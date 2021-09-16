import torch

from GetAnnotation import GetAnnotationABC, get_annotation_fn
from LoadDataset import LoadDatasetABC, load_dataset_base, load_dataset_fn
from Augmentation import AugmentationABC, augmentation_fn
import segmentation_models_pytorch as smp


def command():
    annotation_files = GetAnnotationImp()()  # アノテーションファイル取得
    augmentations = AugmentationImp()()  # データ拡張方法の定義
    dataset = LoadDatasetImp()(annotation_files, augmentations)  # データ読み込み・前処理
    model = build_model_task()  # モデル構築
    loss_func = get_loss_func_task()  # 損失関数
    metrics = get_metrics_task()  # 評価指標
    model_exec_task(model, loss_func, metrics, dataset)  # 学習


class GetAnnotationImp(GetAnnotationABC):
    target = "/mnt/cloudy_z/src/yishikawa/input/Images/Ureter/train_test_cross_val/external/positive"
    img_dir = "movieframe"
    label_dir = "label"
    train_dirs = ["cv1"]
    val_dirs = ["cv2"]
    get_train_fn = get_annotation_fn.seg_case_first_groups
    get_val_fn = get_annotation_fn.seg_case_first_groups

    """
    @classmethod
    def __call__(cls):
        # you can override get_annotation function
        # use class variable, cls.target, cls.img_dir, cls.label_dir, etc..
        return train_set, validation_set
    """


class AugmentationImp(AugmentationABC):
    augmentations = [
        albu.HorizontalFlip(p=0.5),
        augmentation_fn.custom_aug
    ]


class LoadDatasetImp(LoadDatasetABC):
    batch_size = 16
    width, height = (640, 320)
    nb_class = 2
    loader = load_dataset_base.LoadDatasetSgm


def build_model_task():
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        activation="softmax",
        in_channels=3,
        classes=2,
    )
    return model


def get_loss_func_task():
    loss_func = smp.utils.losses.DiceLoss(ignore_channels=[0])
    return loss_func


def get_metrics_task():
    metrics = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0])
    return metrics


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


if __name__ == "__main__":
    command()
