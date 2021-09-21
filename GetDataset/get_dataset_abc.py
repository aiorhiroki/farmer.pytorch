from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List


class GetDatasetSgmABC(Dataset):
    class_values: List[int]

    def __init__(self, annotation, augmentation):
        self.annotation = annotation
        self.augmentation = augmentation

    def __getitem__(self, i):
        img_file, label_file = self.annotation[i]

        image = np.array(Image.open(str(img_file)))
        mask = np.array(Image.open(str(label_file)))

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
