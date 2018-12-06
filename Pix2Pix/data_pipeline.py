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

        if self.opt.is_train:
            self.target_path_list = sorted(glob.glob(os.path.join(self.opt.dataset_dir, 'Target', '*.' + self.opt.dataset_format)))

    def get_transform(self, normalize=True):
        transform_list = []

        if self.opt.is_train and self.opt.flip:
            coin = random.random() > 0.5
            if coin:
                transform_list.append(transforms.Lambda(lambda x: self.__flip(x)))

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    @staticmethod
    def __flip(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    def __getitem__(self, index):
        transform = self.get_transform()

        input_path = self.input_path_list[index]
        input_array = Image.open(input_path)
        input_tensor = transform(input_array)

        if self.opt.is_train:
            target_path = self.target_path_list[index]
            target_array = Image.open(target_path).convert(self.opt.color_space)
            target_tensor = transform(target_array)

            return input_tensor, target_tensor

        else:
            return input_tensor

    def __len__(self):
        return len(self.input_path_list)
