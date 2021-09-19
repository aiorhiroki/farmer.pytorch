import torch


class TrainABC:
    batch_size: int
    epochs: int
    lr: float
    gpu: int
    optim_obj: torch.optim.Optimizer

    def __init__(self, model, loss_func, metrics, train_dataset, val_dataset):
        self.model = model
        self.loss_func = loss_func
        self.metrics = metrics
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def __call__(self):

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False)

        device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.cuda()

        self.optimizer = self.optim_obj(
            [dict(params=self.model.parameters(), lr=self.lr)])

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 1):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 0:
                    print(epoch + 1, i, f"loss: {running_loss / 10}")
                    running_loss = 0.0

            # validation step
            total_loss, total_iou = 0, 0
            with torch.no_grad():
                for (inputs, labels) in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    total_loss += self.loss_func(outputs, labels).item()
                    total_iou += self.metrics(outputs, labels).item()
            mean_loss = total_loss / len(valid_loader.dataset)
            mean_iou = total_iou / len(valid_loader.dataset)
            print("Epoch: {epoch+1}")
            print(f"mean_loss: {mean_loss}, mean_iou: {mean_iou}")

            model_path = f'model_epoch{epoch}.pth'
            torch.save(self.model.state_dict(), model_path)

        print('Finished Training')
