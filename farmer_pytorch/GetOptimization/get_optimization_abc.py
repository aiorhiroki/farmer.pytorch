import torch
from pathlib import Path
from .get_optimization_fn import Logger


class GetOptimizationABC:
    batch_size: int
    epochs: int
    lr: float
    gpu: int
    optim_obj: torch.optim.Optimizer

    def __init__(self, model, loss_func, metrics_func, train_data, val_data):
        self.model = model
        self.loss_func = loss_func
        self.metrics_func = metrics_func
        self.train_data = train_data
        self.val_data = val_data
        self.logger = Logger()

    def __call__(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False)

        device_name = f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(device)
        self.model.to(device)
        self.optimizer = self.optim_obj(
            [dict(params=self.model.parameters(), lr=self.lr)])

        save_model_dir = Path("./models")
        save_model_dir.mkdir(exist_ok=True)

        self.logger.set_metrics(["dice"])
        for epoch in range(self.epochs):
            self.train(train_loader, device, epoch)
            self.validation(valid_loader, device)
            model_path = f'{save_model_dir}/model_epoch{epoch}.pth'
            torch.save(self.model.state_dict(), model_path)
            self.logger.plot_logs()
            self.on_epoch_end()

        return self.logger.get_latest_metrics()

    def train(self, train_loader, device, epoch):
        print(f"\ntrain step, epoch: {epoch + 1}/{self.epochs}")
        self.model.train()
        self.logger.set_progbar(len(train_loader))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            metrics = self.metrics_func(outputs, labels)
            self.logger.get_progbar(loss.item(), metrics.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validation(self, valid_loader, device):
        print("\nvalidation step")
        self.model.eval()
        self.logger.set_progbar(len(valid_loader))
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                metrics = self.metrics_func(outputs, labels)
                self.logger.get_progbar(loss.item(), metrics.item())
        self.logger.update_metrics()

    def on_epoch_end(self):
        pass
