from torch.utils.data import Dataset
from typing import List, Callable
import cv2


class GetDatasetSgmABC(Dataset):
    width: int
    height: int
    nb_class: int


    def __init__(self, annotations, augmentation):
        self.annotations = annotation
        self.augmentation = augmentation

    def __getitem__(self, i):
        img_file, label_file = self.annotations[i]

        image = cv2.imread(img_file)
        mask = cv2.imread(label_file, 0)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # custom preprocessing
        mask[mask == 206] = 1
        mask[mask == 209] = 1
        mask[mask > 1] = 0

        # preprocess image for input
        image = cv2.resize(image, (self.width, self.height)) / 255.
        image = image.transpose(2, 0, 1).astype('float32')
    
        # resize and onehot for mask
        label = np.zeros((self.nb_class, self.height, self.width))
        for class_id in range(self.nb_class):
            class_mask = np.array(mask == class_id, dtype=np.uint8)
            label[class_id] = np.resize(class_mask, (self.width, self.height))

        return image, label

    def __len__(self):
        return len(self.annotations)
