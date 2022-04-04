import torch
import os
from .logger import Logger


class GetOptimizationABC:
    batch_size: int
    epochs: int
    lr: float
    gpus: str
    optimizer: torch.optim.Optimizer
    model: torch.nn.Module
    loss_func: torch.nn.Module
    metric_func: torch.nn.Module
    result_dir: str = 'result'

    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.logger = Logger(self.result_dir)
        self.world_size = len(self.gpus.split(","))
        self.is_distributed = self.world_size > 1
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus

    def __call__(self):
        torch.multiprocessing.spawn(
            self.fit, args=(), nprocs=self.world_size, join=True)

    def fit(self, rank):
        self.setup(rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_data) if self.is_distributed else None
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_data) if self.is_distributed else None
        valid_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size,
            shuffle=False, sampler=valid_sampler)
        self.gpus = self.gpus if torch.cuda.is_available() else []
        self.model.to(rank)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank], find_unused_parameters=True)
        self.optimize = self.optimizer(
            [dict(params=self.model.parameters(), lr=self.lr)])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimize, lr_lambda=self.scheduler_func)

        self.logger.set_metrics(["dice"])
        for epoch in range(self.epochs):
            if self.is_distributed:
                train_sampler.set_epoch(epoch)
            self.train(train_loader, rank, epoch)
            self.validation(valid_loader, rank)
            self.logger.on_epoch_end()
            self.on_epoch_end()

        self.cleanup()

        return self.logger.get_latest_metrics()

    def train(self, train_loader, rank, epoch):
        if rank == 0:
            print(f"\ntrain step, epoch: {epoch + 1}/{self.epochs}")
        self.model.train()
        getattr(self.metric_func, "reset_state", lambda: None)()
        self.logger.set_progbar(len(train_loader))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            metrics = self.metric_func(outputs, labels)
            self.logger.get_progbar(
                loss.item(), metrics.item(), self.scheduler.get_last_lr())
            self.optimize.zero_grad()
            loss.backward()
            self.optimize.step()
        self.scheduler.step()
        if rank == 0:
            torch.save(self.model.state_dict(), f'{self.result_dir}/last.pth')

    def validation(self, valid_loader, rank):
        print("\nvalidation step")
        self.model.eval()
        getattr(self.metric_func, "reset_state", lambda: None)()
        self.logger.set_progbar(len(valid_loader))
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                metrics = self.metric_func(outputs, labels)
                self.logger.get_progbar(loss.item(), metrics.item())
        self.logger.update_metrics()

    def on_epoch_end(self):
        pass

    def setup(self, rank):
        print(f"rank: {rank}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12346'
        torch.distributed.init_process_group(
            "gloo", rank=rank, world_size=self.world_size)

    def cleanup(self):
        torch.distributed.destroy_process_group()

    @staticmethod
    def scheduler_func(epoch):
        if epoch > 10:
            return 0.9 ** (epoch-10)
        else:
            return 1
