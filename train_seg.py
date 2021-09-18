from GetAnnotation import GetAnnotationABC, get_annotation_fn
from LoadDataset import LoadDatasetABC, load_dataset_base, load_dataset_fn
from Augmentation import AugmentationABC, augmentation_fn
import segmentation_models_pytorch as smp
import albumentations as albu


def command():
    annotation_files = GetAnnotationImp()()  # アノテーションファイル取得

    model = get_model_task()  # モデル構築
    loss_func = get_loss_func_task()  # 損失関数
    metrics = get_metrics_task()  # 評価指標
    augmentations = get_augmentation_task()  # データ拡張方法の定義

    dataset = LoadDatasetImp()(annotation_files, augmentations)  # データ読み込み・前処理
    Train()(model, loss_func, metrics, dataset)  # 学習


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


class LoadDatasetImp(LoadDatasetABC):
    batch_size = 16
    width, height = (640, 320)
    nb_class = 2
    loader = load_dataset_base.LoadDatasetSgm


class TrainImp(TrainABC):
    epoch = 16
    

if __name__ == "__main__":
    command()
