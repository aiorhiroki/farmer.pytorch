from typing import List, Callable
from .readers import cross_val


class GetAnnotationABC:
    target: str

    # for train annotation
    img_dir_train: str
    label_dir_train: str
    train_dirs: List[str] = []
    get_train_fn: Callable[[str, str, str, List[str]], List[List[str]]] = None

    # for val annotation
    img_dir_val: str = None
    label_dir_val: str = None
    val_dirs: List[str] = []
    get_val_fn: Callable[[str, str, str, List[str]], List[List[str]]] = None

    # for cross validation
    cv_fold: int = None
    depth: int = 0

    @classmethod
    def __call__(cls):
        if cls.cv_fold:
            return cross_val(cls.get_train_anno(), cls.cv_fold, cls.depth)
        else:
            return (
                None if cls.get_train_fn is None else cls.get_train_anno(),
                None if cls.get_val_fn is None else cls.get_val_anno())

    @classmethod
    def get_train_anno(cls):
        return cls.get_train_fn(
            cls.target, cls.img_dir_train, cls.label_dir_train, cls.train_dirs)

    @classmethod
    def get_val_anno(cls):
        return cls.get_val_fn(
            cls.target, cls.img_dir_val or cls.img_dir_train,
            cls.label_dir_val or cls.label_dir_train, cls.val_dirs)
