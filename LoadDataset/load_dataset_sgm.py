from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class LoadDatasetSgm(Dataset):
    annotations
    preprocessing
    augmentation

    def __getitem__(self, i):
        img_file, label_file = self.annotations[i]

        image = cv2.imread(img_file)
        mask = cv2.imread(label_file, 0)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask 

    def __len__(self):
        return len(self.annotations)
