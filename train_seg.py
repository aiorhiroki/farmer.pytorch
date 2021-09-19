import segmentation_models_pytorch as smp
import albumentations as albu
import torch

from GetAnnotation import GetAnnotationABC, get_annotation_fn
from GetDataset import GetDatasetSgmABC
from Train import TrainABC


def command():
    train_anno, val_anno = GetAnnotationImp()()  # アノテーションファイル取得

    model = get_model_task()  # モデル構築
    loss_func = get_loss_func_task()  # 損失関数
    metrics = get_metrics_task()  # 評価指標
    augmentation = get_augmentation_task()  # データ拡張方法の定義

    train_dataset = DatasetImp(train_anno, augmentation)  # データ読み込み・前処理
    val_dataset = DatasetImp(val_anno, augmentation)

    TrainImp(model, loss_func, metrics, train_dataset, val_dataset)()  # 学習


class GetAnnotationImp(GetAnnotationABC):
    target = "/mnt/cloudy_z/src/yishikawa/input/Images/Ureter/ \
              train_test_cross_val/external/positive"
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


def get_model_task():
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


def get_augmentation_task():
    transforms = [
        albu.HorizontalFlip(p=0.5),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
    ]

    return albu.Compose(transforms)


class DatasetImp(GetDatasetSgmABC):
    width = 640
    height = 320
    nb_class = 2

    """
    def __getitem__(self, i):
        # you can override getitem function
        # use variable, self.annotation, self.augmentation, self.width, etc...
        return in, out
    """


class TrainImp(TrainABC):
    batch_size = 16
    epochs = 16
    lr = 0.001
    gpu = 0
    optimizer = torch.optim.Adam


if __name__ == "__main__":
    command()
