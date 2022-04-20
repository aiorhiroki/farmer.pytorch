from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import dataclasses
from typing import List, Any


@dataclasses.dataclass
class GetDatasetSgm(Dataset):
    annotation: List[List[str]]
    class_values: List[int]
    augmentation: List[Any] = dataclasses.field(default_factory=list)

    def __getitem__(self, i):
        img_file, label_file = self.annotation[i]

        image = np.array(Image.open(img_file))
        mask = np.array(Image.open(label_file))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # custom preprocessing
        image, mask = self.preprocess(image, mask)

        # preprocess image for input
        image = image.transpose(2, 0, 1).astype('float32') / 255.

        masks = [(mask == v) for v in self.class_values]
        mask = np.array(masks, dtype='float32')

        return image, mask

    def __len__(self):
        return len(self.annotation)

    def preprocess(self, image, mask):
        return image, mask
