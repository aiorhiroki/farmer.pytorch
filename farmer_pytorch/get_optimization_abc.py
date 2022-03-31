import torch
from typing import Callable
from .logger import Logger


class GetOptimizationABC:
    batch_size: int
    epochs: int
    lr: float
    gpu: int
    optimizer: torch.optim.Optimizer
    scheduler_func: Callable[int, float]
    model: torch.nn.Module
    loss_func: torch.nn.Module
    metric_func: torch.nn.Module
    result_dir: str = 'result'

    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.logger = Logger(self.result_dir)

    def __call__(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False)

        device_name = f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(device)
        self.model.to(device)
        self.optimize = self.optimizer(
            [dict(params=self.model.parameters(), lr=self.lr)])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimize, lr_lambda=self.scheduler_func)

        self.logger.set_metrics(["dice"])
        for epoch in range(self.epochs):
            self.train(train_loader, device, epoch)
            self.validation(valid_loader, device)
            torch.save(self.model.state_dict(), f'{self.result_dir}/last.pth')
            self.logger.on_epoch_end()
            self.on_epoch_end()

        return self.logger.get_latest_metrics()

    def train(self, train_loader, device, epoch):
        print(f"\ntrain step, epoch: {epoch + 1}/{self.epochs}")
        self.model.train()
        getattr(self.metric_func, "reset_state", lambda: None)()
        self.logger.set_progbar(len(train_loader))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            metrics = self.metric_func(outputs, labels)
            self.logger.get_progbar(loss.item(), metrics.item())
            self.optimize.zero_grad()
            loss.backward()
            self.optimize.step()
            self.scheduler.step()

    def validation(self, valid_loader, device):
        print("\nvalidation step")
        self.model.eval()
        getattr(self.metric_func, "reset_state", lambda: None)()
        self.logger.set_progbar(len(valid_loader))
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                metrics = self.metric_func(outputs, labels)
                self.logger.get_progbar(loss.item(), metrics.item())
        self.logger.update_metrics()

    def on_epoch_end(self):
        pass
