import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import os
import cv2
import albumentations as A
from module.datasets.imagenet import ImageNet
from module.datasets.intelscene import IntelScene
from tqdm import tqdm

def change_color(img, o1, o2, o3):
    ch1 = img[:,:,o1:o1+1]
    ch2 = img[:,:,o2:o2+1]
    ch3 = img[:,:,o3:o3+1]
    return np.concatenate([ch1, ch2, ch3], -1)

class DataLoader:
    def __init__(self, data_path='~/Datasets', goal='imagenet', source='imagenet', width=224, height=224, debug=False):
        self.width, self.height = width, height
        self.goal = goal
        self.source = source

        if goal == 'imagenet':
            goal_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(size=(height, width)),
            ])
            # ImageNet 2012 train/val
            # 1,281,167 pics
            self.goal_trainset = ImageNet(root=os.path.expanduser(f'{data_path}/imagenet'), transform=goal_transform, split='train')
            # 50,000 pics
            self.goal_evalset = ImageNet(root=os.path.expanduser(f'{data_path}/imagenet'), transform=goal_transform, split='val')
        elif goal == 'mnist':
            goal_transform = transforms.Compose([
                transforms.Resize((height, width)),
            ])
            self.goal_trainset = torchvision.datasets.MNIST(root=os.path.expanduser(data_path), transform=goal_transform, train=True)
            self.goal_evalset = torchvision.datasets.MNIST(root=os.path.expanduser(data_path), transform=goal_transform, train=False)
        elif goal == 'flower':
            goal_transform = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(size=(224, 224)),
                transforms.Resize((height, width)),
            ])
            self.goal_trainset = torchvision.datasets.Flowers102(root=os.path.expanduser(data_path), transform=goal_transform, split='train')
            self.goal_evalset = torchvision.datasets.Flowers102(root=os.path.expanduser(data_path), transform=goal_transform, split='val')
        elif goal == 'celeba':
            goal_transform = transforms.Compose([
                transforms.CenterCrop(size=(128, 128)),
                transforms.Resize((height, width)),
            ])
            self.goal_trainset = torchvision.datasets.CelebA(root=os.path.expanduser(data_path), transform=goal_transform, split='train')
            self.goal_evalset = torchvision.datasets.CelebA(root=os.path.expanduser(data_path), transform=goal_transform, split='valid')
        elif goal == 'scene':
            goal_transform = transforms.Compose([
                transforms.Resize((height, width)),
            ])
            self.goal_trainset = IntelScene(root=os.path.expanduser(data_path), transform=goal_transform, split='train')
            self.goal_evalset = IntelScene(root=os.path.expanduser(data_path), transform=goal_transform, split='val')

        if source == 'imagenet':
            source_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(size=(height, width)),
            ])
            # ImageNet 2012 train/val
            # 1,281,167 pics
            self.source_trainset = ImageNet(root=os.path.expanduser(f'{data_path}/imagenet'), transform=source_transform, split='train')
            # 50,000 pics
            self.source_evalset = ImageNet(root=os.path.expanduser(f'{data_path}/imagenet'), transform=source_transform, split='val')
        elif source == 'dtd':
            source_transform = A.Compose([
                A.RandomCrop(height=height, width=width),
            ])
            self.source_trainset = []
            self.source_evalset = []
            print('Loading DTD sources')
            if debug:
                labels = os.listdir(os.path.expanduser(f'{data_path}/dtd/images'))[:5]
            else:
                labels = os.listdir(os.path.expanduser(f'{data_path}/dtd/images'))
            for label in tqdm(labels):
                img_paths = os.listdir(os.path.expanduser(f'{data_path}/dtd/images/{label}/'))
                for i, img in enumerate(img_paths):
                    source = cv2.imread(os.path.expanduser(f'{data_path}/dtd/images/{label}/{img}'))
                    if source is not None:
                        image = source_transform(image=source)['image']
                        image2 = change_color(image, 0, 2, 1)
                        image3 = change_color(image, 1, 0, 2)
                        image4 = change_color(image, 1, 2, 0)
                        image5 = change_color(image, 2, 0, 1)
                        image6 = change_color(image, 2, 1, 0)
                        if i < len(img_paths)*0.8:
                            self.source_trainset.append((image, label))
                            self.source_trainset.append((image2, label))
                            self.source_trainset.append((image3, label))
                            self.source_trainset.append((image4, label))
                            self.source_trainset.append((image5, label))
                            self.source_trainset.append((image6, label))
                        else:
                            self.source_evalset.append((image, label))
                            self.source_evalset.append((image2, label))
                            self.source_evalset.append((image3, label))
                            self.source_evalset.append((image4, label))
                            self.source_evalset.append((image5, label))
                            self.source_evalset.append((image6, label))
        elif source == goal:
            self.source_trainset = self.goal_trainset
            self.source_evalset = self.goal_evalset


    def get_random_goal(self, mode):
        if mode == 'train':
            dataset = self.goal_trainset
        elif mode == 'eval':
            dataset = self.goal_evalset
        goal_idx = random.randint(0, len(dataset)-1)
        image, label = dataset[goal_idx]  # PIL, int
        image = np.array(image)/255
        return image, label, goal_idx

    def get_random_source(self, mode):
        if mode == 'train':
            dataset = self.source_trainset
        elif mode == 'eval':
            dataset = self.source_evalset
        image, label = random.choice(dataset)  # PIL, int
        image = np.array(image)/255
        return image, label

    def get_random_source_pool(self, n, mode):
        if self.goal == self.source:
            return self.get_random_goals(n, mode), None

        if mode == 'train':
            dataset = self.source_trainset
        elif mode == 'eval':
            dataset = self.source_evalset

        images = []
        labels = []
        source_ids = np.array([i for i in range(len(dataset))])
        sampled_ids = np.random.choice(source_ids, n, replace=False)
        for sample_id in sampled_ids:
            image, label = dataset[sample_id]
            images.append(image/255)
            labels.append(label)

        return np.stack(images), np.stack(labels)

    def get_random_goals(self, n, mode):
        if mode == 'train':
            dataset = self.goal_trainset
        elif mode == 'eval':
            dataset = self.goal_evalset
        goals = []
        if self.goal in ['celeba_single', 'flower_single']:
            sampled_ids = np.array([self.selected_goal_idx for i in range(n)])
        else:
            sampled_ids = np.random.choice(np.arange(len(dataset)), n, replace=False)
        for data_idx in sampled_ids:
            # if self.goal == 'imagenet':
            #     data, _, _ = dataset[data_idx.item()]
            if self.goal == 'mnist':
                data, _ = dataset[data_idx.item()]
                data = np.expand_dims(data, -1).repeat(3, -1)  # Grayscale to RGB
                data = 255 - data  # White to black (dark background to light background)
            elif self.goal in ['imagenet', 'flower', 'celeba', 'celeba_single', 'flower_single']:
                data, _ = dataset[data_idx.item()]
            elif self.goal in ['scene']:
                data = dataset[data_idx.item()]
            data = np.array(data)/255
            goals.append(data)
        
        return np.stack(goals)

    def get_random_sources(self, n, mode):
        if self.goal == self.source:
            return self.get_random_goals(n, mode)

        if mode == 'train':
            dataset = self.source_trainset
        elif mode == 'eval':
            dataset = self.source_evalset
        images = []
        sampled_ids = np.random.choice(np.arange(len(dataset)), n, replace=False)
        for sample_id in sampled_ids:
            image, _ = dataset[sample_id]
            images.append(image/255)

        return np.stack(images)