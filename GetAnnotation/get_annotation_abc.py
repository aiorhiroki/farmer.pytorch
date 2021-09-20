from typing import List, Callable


class GetAnnotationABC:
    target: str
    img_dir: str
    label_dir: str

    train_dirs: List[str]
    val_dirs: List[str]

    get_train_fn: Callable[[str, str, str, List[str]], List[List[str]]]
    get_val_fn: Callable[[str, str, str, List[str]], List[List[str]]]

    @classmethod
    def __call__(cls):
        train_set = cls.get_train_fn(
            cls.target, cls.img_dir, cls.label_dir, cls.train_dirs
        )
        validation_set = cls.get_val_fn(
            cls.target, cls.img_dir, cls.label_dir, cls.val_dirs
        )
        print("completed")
        return train_set, validation_set

    def __init__(self):
        print("get annotation..."+" "*10, end="")
