from GetAnnotation import GetAnnotationABC, get_annotation_fn
from GetDataset import GetDatasetSgmABC
from GetOptimization import GetOptimizationABC

import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import cv2


def command():
    train_anno, val_anno = GetAnnotationImp()()  # アノテーションファイル取得

    model = get_model_task()  # モデル構築
    loss_func = get_loss_func_task()  # 損失関数
    metrics = get_metrics_task()  # 評価指標
    train_aug, val_aug = get_augmentation_task()  # データ拡張方法の定義
    # データ読み込み・前処理
    train_dataset = DatasetImp(train_anno, train_aug)
    val_dataset = DatasetImp(val_anno, val_aug)
    # 学習
    GetOptimizationImp(model, loss_func, metrics, train_dataset, val_dataset)()


class GetAnnotationImp(GetAnnotationABC):
    target = "./seg_data"

    img_dir_train = "train"
    label_dir_train = "trainannot"
    get_train_fn = get_annotation_fn.seg_case_direct

    img_dir_val = "val"
    label_dir_val = "valannot"
    get_val_fn = get_annotation_fn.seg_case_direct

    """
    @classmethod
    def __call__(cls):
        # you can override GetAnnotation function
        # use class variable, cls.target, cls.img_dir, cls.label_dir, etc..
        return train_set, validation_set
    """


def get_model_task():
    model = smp.FPN(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        activation="sigmoid",
        in_channels=3,
        classes=1,
    )
    return model


def get_loss_func_task():
    loss_func = smp.utils.losses.DiceLoss()
    return loss_func


def get_metrics_task():
    metrics = smp.utils.metrics.Fscore(threshold=0.5)
    return metrics


def get_augmentation_task():
    train_transform = [
        albu.augmentations.transforms.HorizontalFlip(p=0.5),
    ]

    val_transform = [
        albu.PadIfNeeded(256, 512)
    ]

    return albu.Compose(train_transform), albu.Compose(val_transform)


class DatasetImp(GetDatasetSgmABC):
    class_values = [100]

    # custom preprocessing
    def preprocess(self, image, mask):
        width = 512
        height = 256
        mask = cv2.resize(mask, (width, height))
        image = cv2.resize(image, (width, height))
        return image, mask

    """
    def __getitem__(self, i):
        # you can override getitem function
        # use instance/class variable, self.annotation, self.augmentation ...
        return in, out
    """


class GetOptimizationImp(GetOptimizationABC):
    batch_size = 8
    epochs = 50
    lr = 0.001
    gpu = 0
    optim_obj = torch.optim.Adam

    """
    def on_epoch_end(self):
        # set custom callbacks
    """


if __name__ == "__main__":
    command()
