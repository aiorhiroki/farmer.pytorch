from GetAnnotation import GetAnnotationABC, get_annotation_fn
from GetDataset import GetDatasetSgmABC
from GetOptimization import GetOptimizationABC

import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import cv2


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
        albu.PadIfNeeded(352, 640)
    ]

    return albu.Compose(train_transform), albu.Compose(val_transform)


class DatasetImp(GetDatasetSgmABC):
    class_values = [1]

    # custom preprocessing
    def preprocess(self, image, mask):
        width = 640
        height = 352
        mask[mask == 206] = 1
        mask[mask == 209] = 1
        mask[mask > 1] = 0
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
    batch_size = 2
    epochs = 50
    lr = 0.001
    gpu = 0
    optim_obj = torch.optim.Adam

    """
    def on_epoch_end(self):
        # set custom callbacks
    """
