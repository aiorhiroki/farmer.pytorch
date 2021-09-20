from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List


class GetDatasetSgmABC(Dataset):
    class_values: List[int]

    def __init__(self, annotation, augmentation):
        self.annotation = annotation
        self.augmentation = augmentation

    def __getitem__(self, i):
        img_file, label_file = self.annotation[i]

        image = cv2.imread(str(img_file))
        mask = cv2.imread(str(label_file), 0)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # custom preprocessing
        image, mask = self.preprocess(image, mask)

        # preprocess image for input
        image = image.transpose(2, 0, 1).astype('float32') / 255.

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        return image, mask

    def __len__(self):
        return len(self.annotation)

    def preprocess(self, image, mask):
        return image, mask
