import os
from os.path import join
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize, ToTensor
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, opt, mode):
        super(CustomDataset, self).__init__()
        dataset_name = opt.dataset_name
        dir_datasets = opt.dir_datasets

        if mode == 'train':
            dir_input = join(dir_datasets, dataset_name, 'Train', 'A')
            dir_real = join(dir_datasets, dataset_name, 'Train', 'B')
            self.list_paths_input, self.list_paths_real = sorted(os.listdir(dir_input)), sorted(os.listdir(dir_real))
            self.dir_input, self.dir_real = dir_input, dir_real

        elif mode == 'val':
            dir_input = join(dir_datasets, dataset_name, 'Test', 'A')
            dir_real = join(dir_datasets, dataset_name, 'Test', 'B')
            self.list_paths_input, self.list_paths_real = sorted(os.listdir(dir_input)), sorted(os.listdir(dir_real))
            self.dir_input, self.dir_real = dir_input, dir_real

        elif mode == 'val_A2B':
            dir_input = join(dir_datasets, dataset_name, 'Test', 'A')
            self.list_paths_input = sorted(os.listdir(dir_input))
            self.dir_input = dir_input

        elif mode == 'val_B2A':
            dir_input = join(dir_datasets, dataset_name, 'Test', 'B')
            self.list_paths_input = sorted(os.listdir(dir_input))
            self.dir_input = dir_input

        elif mode == 'test':
            dir_input = join(dir_datasets, dataset_name, 'Test', 'A')
            self.list_paths_input = sorted(os.listdir(dir_input))
            self.dir_input = dir_input

        else:
            raise NotImplemented("Invalid mode {}. Choose among 'train', 'val', 'val_A2B', 'val_B2A' and 'test'."
                                 .format(mode))

        self.dataset_name = opt.dataset_name
        self.load_size = opt.load_size
        self.mode = mode

    def __getitem__(self, index):
        transforms = list()

        if self.dataset_name == 'facades':
            transforms += [Resize((self.load_size, self.load_size), Image.BILINEAR)]
            transforms += [RandomHorizontalFlip()] if random.random() > 0.5 else []
            transforms += [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

            image_input = Image.open(join(self.dir_input, self.list_paths_input[index])).convert('RGB')

            if self.mode == 'train' or self.mode == 'val':
                index_random = random.randint(0, len(self.list_paths_real) - 1)
                image_real = Image.open(join(self.dir_real, self.list_paths_real[index_random])).convert('RGB')

        else:
            transforms += [Resize((self.load_size, self.load_size), Image.BILINEAR)]
            transforms += [ToTensor(), Normalize(mean=[0.5], std=[0.5])]

            image_input = Image.open(join(self.dir_input, self.list_paths_input[index])).convert('L')

            if self.mode == 'train' or self.mode == 'val':
                index_random = random.randint(0, len(self.list_paths_real) - 1)
                image_real = Image.open(join(self.dir_real, self.list_paths_real[index_random])).convert('L')

        transforms = Compose(transforms)
        input = transforms(image_input)
        real = transforms(image_real) if self.mode == 'train' or self.mode == 'val' else 0

        return input, real

    def __len__(self):
        return len(self.list_paths_input)
