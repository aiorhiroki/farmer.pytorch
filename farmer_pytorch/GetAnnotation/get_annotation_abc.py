from typing import List, Callable


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
    get_val_fn: Callable[[str, str, str, List[str]], List[List[str]]]

    @classmethod
    def __call__(cls):
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
