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
            # train and validation
            loss, metrics = self.train(train_loader, device, epoch)
            val_loss, val_metrics = self.validation(valid_loader, device)

            # update metrics plot
            self.logger.plot_metrics(val_metrics, "dice")

            # save result
            model_path = f'{save_model_dir}/model_epoch{epoch}.pth'
            torch.save(self.model.state_dict(), model_path)

            self.on_epoch_end()  # custom callbacks

        print('\nFinished Training')

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
        return loss.item(), metrics.item()


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
        return loss.item(), metrics.item()

    def on_epoch_end(self):
        pass
