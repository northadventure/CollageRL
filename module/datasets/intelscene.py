import os
from PIL import Image
from torch.utils.data import Dataset

class IntelScene(Dataset):
    def __init__(self, root, transform, split):
        self.root = root
        self.transform = transform
        self.split = split
        if split == 'train':
            self.image_list = os.listdir(f'{root}/IntelScene/train')
        else:
            self.image_list = os.listdir(f'{root}/IntelScene/val')

    def __len__(self):
        if self.split == 'train':
            return 21335
        else:
            return 3000

    def __getitem__(self, idx):
        if self.split == 'train':
            image = Image.open(f'{self.root}/IntelScene/train/{self.image_list[idx]}')
        else:
            image = Image.open(f'{self.root}/IntelScene/val/{self.image_list[idx]}')
        if self.transform:
            image = self.transform(image)

        return image