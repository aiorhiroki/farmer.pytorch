from typing import List, Callable
from .get_annotation_fn import crossval


class GetAnnotationABC:
    target: str

    # for train annotation
    img_dir_train: str
    label_dir_train: str
    train_dirs: List[str] = None
    get_train_fn: Callable[[str, str, str, List[str]], List[List[str]]]

    # for val annotation
    img_dir_val: str = None
    label_dir_val: str = None
    val_dirs: List[str] = None
    get_val_fn: Callable[[str, str, str, List[str]], List[List[str]]] = None

    # for cross validation
    cv_fold: int = None
    depth: int = 0

    @classmethod
    def __call__(cls):
        if cls.cv_fold:
            return crossval(cls.get_train_annos(), cls.cv_fold, cls.depth)
        else:
            return [cls.get_train_annos()], [cls.get_val_annos()]

    @classmethod
    def get_train_annos(cls):
        return cls.get_train_fn(
            cls.target, cls.img_dir_train, cls.label_dir_train, cls.train_dirs)

    @classmethod
    def get_val_annos(cls):
        return cls.get_val_fn(
            cls.target, cls.img_dir_val or cls.img_dir_train,
            cls.label_dir_val or cls.label_dir_train, cls.val_dirs)
