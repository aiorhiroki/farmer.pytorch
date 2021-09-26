from farmer_pytorch.GetAnnotation import GetAnnotationABC, get_annotation_fn
from farmer_pytorch.GetDataset import GetDatasetSgmABC
from farmer_pytorch.GetOptimization import GetOptimizationABC

import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import numpy as np
import cv2


class GetAnnotationImp(GetAnnotationABC):
    target = "./seg_data/CamVid"

    img_dir_train = "train"
    label_dir_train = "trainannot"
    get_train_fn = get_annotation_fn.seg_case_direct

    img_dir_val = "val"
    label_dir_val = "valannot"
    get_val_fn = get_annotation_fn.seg_case_direct

    cv_fold = 5

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
    class_values = [8]

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
    epochs = 10
    lr = 0.001
    gpu = 0
    optim_obj = torch.optim.Adam

    """
    def on_epoch_end(self):
        # set custom callbacks
    """


def command():
    mean_dice = list()
    # アノテーションファイル取得
    train_annos, val_annos = GetAnnotationImp()()
    for i, (train_anno, val_anno) in enumerate(zip(train_annos, val_annos), 1):
        print(f"trial: {i}/{len(train_annos)}")

        # モデル構築
        model = get_model_task()
        # 損失関数
        loss_fn = get_loss_func_task()
        # 評価指標
        metrics = get_metrics_task()
        # データ拡張方法の定義
        train_aug, val_aug = get_augmentation_task()
        # データ読み込み・前処理
        train_data = DatasetImp(train_anno, train_aug)
        val_data = DatasetImp(val_anno, val_aug)
        # 学習
        d = GetOptimizationImp(model, loss_fn, metrics, train_data, val_data)()

        mean_dice.append(d)
    print("mean_dice: ", np.mean(mean_dice))


if __name__ == "__main__":
    command()
