from GetAnnotation import GetAnnotationABC, get_annotation_fn
from GetDataset import GetDatasetSgmABC
from GetOptimization import GetOptimizationABC

import segmentation_models_pytorch as smp
import albumentations as albu
import torch


class GetAnnotationImp(GetAnnotationABC):
    target = "/mnt/cloudy_z/src/yishikawa/input/"
    target += "Images/Ureter/train_test_cross_val/external/positive"
    img_dir = "movieframe"
    label_dir = "label"
    train_dirs = ["cv1"]
    val_dirs = ["cv2"]
    get_train_fn = get_annotation_fn.seg_case_first_groups
    get_val_fn = get_annotation_fn.seg_case_first_groups

    """
    @classmethod
    def __call__(cls):
        # you can override GetAnnotation function
        # use class variable, cls.target, cls.img_dir, cls.label_dir, etc..
        return train_set, validation_set
    """


def get_model_task():
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        activation="softmax2d",
        in_channels=3,
        classes=2,
    )
    return model


def get_loss_func_task():
    loss_func = smp.utils.losses.DiceLoss()
    return loss_func


def get_metrics_task():
    metrics = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0])
    return metrics


def get_augmentation_task():
    transforms = [
        albu.augmentations.transforms.HorizontalFlip(p=0.5),
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


class GetOptimizationImp(GetOptimizationABC):
    batch_size = 4
    epochs = 16
    lr = 0.001
    gpu = 0
    optim_obj = torch.optim.Adam
