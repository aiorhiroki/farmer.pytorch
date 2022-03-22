from farmer_pytorch.GetAnnotation import GetAnnotationABC, get_annotation_fn
from farmer_pytorch.GetDataset import GetDatasetSgmABC
from farmer_pytorch.GetOptimization import GetOptimizationABC, optimization_fn

import segmentation_models_pytorch as smp
import albumentations as albu
import torch


class GetAnnotationImp(GetAnnotationABC):
    target = "./seg_data/CamVid"

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


class DatasetImp(GetDatasetSgmABC):
    class_values = [8]
    train_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512),
         albu.augmentations.transforms.HorizontalFlip(p=0.5)])
    val_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512)])

    """
    def preprocess(self, image, mask):
        # custom preprocessing
    """

    """
    def __getitem__(self, i):
        # you can override getitem function
        # use instance/class variable, self.annotation, self.augmentation ...
        return in, out
    """


class GetOptimizationImp(GetOptimizationABC):
    batch_size = 16
    epochs = 10
    lr = 0.001
    gpu = 0
    optim_obj = torch.optim.Adam
    model = smp.FPN(encoder_name="efficientnet-b7", encoder_weights="imagenet",
                    activation="sigmoid", in_channels=3, classes=1,)
    loss_func = smp.losses.DiceLoss('binary')
    metric_obj = optimization_fn.Fscore

    """
    def on_epoch_end(self):
        # set custom callbacks
    """


def command():
    train_anno, val_anno = GetAnnotationImp()()
    train, val = DatasetImp(train_anno, training=True), DatasetImp(val_anno)
    GetOptimizationImp(train, val)()


if __name__ == "__main__":
    command()
