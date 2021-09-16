from dataclasses import dataclass
from Augmentation import augmentation_fn
from typing import List, Callable


@dataclass(init=False)
class AugmentationABC:
    hoge: str
