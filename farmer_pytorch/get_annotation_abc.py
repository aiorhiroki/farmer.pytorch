from typing import List, Any
from .readers import cross_val
import dataclasses


@dataclasses.dataclass
class GetAnnotationABC:
    target: str

    # for train annotation
    train_dirs: List[str] = dataclasses.field(default_factory=list)
    get_train_fn: Any = None

    # for val annotation
    val_dirs: List[str] = dataclasses.field(default_factory=list)
    get_val_fn: Any = None

    # for cross validation
    cv_fold: int = None
    depth: int = 0

    def __call__(self):
        if self.cv_fold:
            return cross_val(self.get_train_anno(), self.cv_fold, self.depth)
        else:
            return (
                None if self.get_train_fn is None else self.get_train_anno(),
                None if self.get_val_fn is None else self.get_val_anno())

    def get_train_anno(self):
        return self.get_train_fn(self.target, self.train_dirs)

    def get_val_anno(self):
        return self.get_val_fn(self.target, self.val_dirs)
