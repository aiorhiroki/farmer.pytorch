import torch

from dataclasses import dataclass
from GetAnnotation import get_annotation_fn
from typing import List, Callable


@dataclass(init=False)
class TrainABC:


def model_exec_task(model, loss_func, metrics, dataset):
    train_loader, valid_loader = dataset
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.cuda()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print(epoch + 1, i + 1, f"loss: {running_loss / 10}")
                running_loss = 0.0

        # validation step
        total_loss, total_iou = 0, 0
        with torch.no_grad():
            for (inputs, labels) in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += loss_func(outputs, labels).item()
                total_iou += metrics(outputs, labels).item()
        mean_loss = total_loss / len(valid_loader.dataset)
        mean_iou = total_iou / len(valid_loader.dataset)
        print("Epoch: {epoch+1}")
        print(f"mean_loss: {mean_loss}, mean_iou: {mean_iou}")

        model_path = f'model_epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)

    print('Finished Training')