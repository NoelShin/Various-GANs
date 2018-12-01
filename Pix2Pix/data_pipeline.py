import os
import glob
import random
import torch
from torchvision import transforms as transforms
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.input_path_list = sorted(glob.glob(os.path.join(self.opt.dataset_dir, 'Input', '*.' + self.opt.dataset_format)))
        self.target_path_list = sorted(glob.glob(os.path.join(self.opt.dataset_dir, 'Target', '*.' + self.opt.dataset_format)))

        if self.opt.flip:
            self.flip = random.random() > 0.5

    def get_transform(self, normalize=True):
        transform_list = []

        if self.opt.is_train and self.opt.flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        input_path = self.input_path_list[index]
        input_array = Image.open(input_path)
        transform = self.get_transform()
        input_tensor = transform(input_array)

        target_path = self.target_path_list[index]
        target_array = Image.open(target_path).convert('RGB')
        target_tensor = transform(target_array)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.input_path_list)
