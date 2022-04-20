import torch
import os
from .logger import Logger
from .metrics import SegMetrics
import dataclasses


@dataclasses.dataclass
class GetOptimizationABC:
    train_data: torch.utils.data.Dataset
    val_data: torch.utils.data.Dataset
    batch_size: int
    epochs: int
    lr: float
    gpus: str
    optimizer_cls: torch.optim.Optimizer
    model: torch.nn.Module
    loss_func: torch.nn.Module
    result_dir: str = 'result'
    port: str = '12346'

    def __post_init__(self):
        self.logger = Logger(self.result_dir)
        self.world_size = len(self.gpus.split(","))
        self.is_distributed = self.world_size > 1

    def __call__(self):
        torch.multiprocessing.spawn(
            self.fit, args=(), nprocs=self.world_size, join=True)

    def fit(self, rank):
        self.set_env(rank)
        self.set_params(rank)
        sampler, train_loader, valid_loader = self.make_data_loader()
        for epoch in range(self.epochs):
            self.train(train_loader, rank, sampler, epoch)
            self.validation(valid_loader, rank)
        self.cleanup()

    def set_params(self, rank):
        self.model = self.model.to(rank)
        self.model_without_ddp = self.model
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank], find_unused_parameters=True)
            self.model_without_ddp = self.model.module
        self.set_optimizer()
        self.set_scheduler()

    def set_optimizer(self):
        self.optimizer = self.optimizer_cls(
            [dict(params=self.model.parameters(), lr=self.lr)])

    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.scheduler_func)

    def make_data_loader(self):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_data) if self.is_distributed else None
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, drop_last=True,
            shuffle=(train_sampler is None), sampler=train_sampler)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_data) if self.is_distributed else None
        valid_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, drop_last=True,
            shuffle=False, sampler=valid_sampler)
        return train_sampler, train_loader, valid_loader

    def train(self, train_loader, rank, sampler, epoch):
        if self.is_distributed:
            sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\ntrain step, epoch: {epoch + 1}/{self.epochs}")
            self.logger.set_progbar(len(train_loader))
        self.model.train()
        for inputs, labels in train_loader:
            outputs = self.model(inputs.to(rank))
            loss = self.loss_func(outputs, labels.to(rank))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if rank == 0:
                self.logger(loss.item(), lr=self.scheduler.get_last_lr())
        self.scheduler.step()
        if rank == 0:
            torch.save(
                self.model_without_ddp.state_dict(),
                f'{self.result_dir}/last.pth')

    def validation(self, valid_loader, rank):
        if rank == 0:
            print("\nvalidation step")
            self.logger.set_progbar(len(valid_loader))
        self.model.eval()
        metrics = SegMetrics()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = self.model(inputs.to(rank))
                loss = self.loss_func(outputs, labels.to(rank))
                confusion = metrics.calc_confusion(outputs, labels.to(rank))
                if self.is_distributed:
                    torch.distributed.all_reduce(confusion)
                if rank == 0:
                    dice = metrics.compute_metric(confusion, metrics.dice)
                    self.logger(loss.item(), dice=dice.item())
        if rank == 0:
            self.logger.update_metrics()

    def set_env(self, rank):
        print(f"rank: {rank}")
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.port
        torch.distributed.init_process_group(
            "gloo", rank=rank, world_size=self.world_size)

    def cleanup(self):
        torch.distributed.destroy_process_group()

    @staticmethod
    def scheduler_func(epoch):
        return 0.95 ** (epoch-10) if epoch > 10 else 1
