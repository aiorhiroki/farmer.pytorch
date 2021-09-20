import torch
from .get_optimization_fn import get_prob_bar


class GetOptimizationABC:
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
            self.model.train()
            running_loss = 0.0
            nb_train_iters = len(train_loader.dataset) // self.batch_size
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
                cout = f"epoch: {epoch + 1}/{self.epochs} "
                cout += get_prob_bar(i, nb_train_iters)
                cout += f" loss: {(running_loss / i):.5g}"
                if i == 0:
                    print(cout)
                else:
                    print("\r"+cout, end="")

            # validation step
            print("\nValidation step starts...")
            self.model.eval()
            total_loss, total_dice = 0, 0
            with torch.no_grad():
                nb_valid_iters = len(valid_loader.dataset) // self.batch_size
                for i, data in enumerate(valid_loader, 1):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    total_loss += self.loss_func(outputs, labels).item()
                    total_dice += self.metrics(outputs, labels).item()

                    cout = get_prob_bar(i, nb_valid_iters)
                    cout += f" loss: {(total_loss / i):.5g}"
                    cout += f" dice: {(total_dice / i):.5g}"
                    print("\r"+cout, end="")

            model_path = f'models/model_epoch{epoch}.pth'
            torch.save(self.model.state_dict(), model_path)

        print('\nFinished Training')
