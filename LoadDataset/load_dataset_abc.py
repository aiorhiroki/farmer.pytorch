from dataclasses import dataclass
from LoadDataset import load_dataset_fn
from typing import List, Callable
from torch.utils.data import DataLoader


@dataclass(init=False)
class LoadDatasetABC:
    batch_size: int
    img_preprocessing_fns = List[]
    label_preprocessing_fns = List[]

    @classmethod
    def __call__(cls, annotation_set):
        print("override load dataset flow")
        train_set, val_set = annotation_set

        train_dataset = Dataset(train_set, preprocessing, augmentation)
        val_dataset = Dataset(val_set, preprocessing, augmentation)

        train_loader = DataLoader(
            train_dataset, batch_size=cls.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            val_dataset, batch_size=cls.batch_size, shuffle=False
        )

        return train_loader, valid_loader

    def __init__(self):
        print("load dataset task")
