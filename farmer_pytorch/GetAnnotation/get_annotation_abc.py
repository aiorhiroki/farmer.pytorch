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
    img_dir_val: str
    label_dir_val: str
    val_dirs: List[str] = None
    get_val_fn: Callable[[str, str, str, List[str]], List[List[str]]] = None

    @classmethod
    def __call__(cls, cv_fold: int = None, cv_i=0, depth=0):
        if cv_fold:
            print(f"cross_val: {cv_i+1}/{cv_fold}")
            return crossval(cls.get_train_annotations(), cv_fold, cv_i, depth)
        else:
            return cls.get_train_annotations(), cls.get_val_annotations()

    @classmethod
    def get_train_annotations(cls):
        return cls.get_train_fn(
            cls.target, cls.img_dir_train, cls.label_dir_train, cls.train_dirs)

    @classmethod
    def get_val_annotations(cls):
        return cls.get_val_fn(
            cls.target, cls.img_dir_val, cls.label_dir_val, cls.val_dirs)

    def __init__(self):
        print("get annotation...")
