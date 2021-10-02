import os
import glob
from pathlib import Path
import multiprocessing as mp
import ssl

import numpy as np
from PIL import Image

import torch
from torchvision.datasets.utils import download_url, extract_archive
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from torchvision import transforms

def download_dataset(url='https://sid.erda.dk/share_redirect/heaAFNnmaG/data.zip',
                     zip_name='beetles.zip', folder_name='beetles',
                     force_download=False, root='./data/'):
    ssl._create_default_https_context = ssl._create_unverified_context
    archive = os.path.join(root, zip_name)
    data_folder = os.path.join(root, folder_name)
    if (not os.path.exists(data_folder) or force_download):
        download_url(url, root, zip_name)
        extract_archive(archive, data_folder, False)
    return data_folder

def image_folder_stats(data_folder):
    dims_list = []

    for filename in glob.iglob(data_folder + '/**/*.jpg', recursive=True):
        im = Image.open(filename)
        dims_list.append(im.size)

    dims_list = np.array(dims_list)
    avg_dims = np.mean(dims_list, axis=0)[::-1]
    min_dims = np.min(dims_list, axis=0)[::-1]
    max_dims = np.max(dims_list, axis=0)[::-1]
    return avg_dims, min_dims, max_dims

def split_dataset(dataset, train_ratio, val_ratio):

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * (len(dataset) - train_size))
    test_size = len(dataset) - (train_size + val_size)
    dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
    train_data, val_data, test_data = random_split(dataset, dataset_sizes.values())
    return train_data, val_data, test_data, dataset_sizes


def dataset_stats(data_set, load_mean_std=True, num_workers=mp.cpu_count(),
                  path='./models/beetle_mean_std.pt', batch_size=32):

    if load_mean_std:
        mean, std = torch.load(path)

    else:
        loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for data, _ in loader:

            b, c, h, w = data.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels
        mean = fst_moment
        std = torch.sqrt(snd_moment - fst_moment ** 2)
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(torch.stack((mean, std)), path)
    return mean, std

class TransformsDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def augmentation_1(start_height, start_width, scale, theta, mean, std):
    img_height_crop, img_width_crop = int(start_height * scale), int(start_width)
    resize = transforms.Resize((start_height, start_width))
    center_crop = transforms.CenterCrop((img_height_crop, img_width_crop))

    rotate = transforms.RandomRotation((theta, theta))
    random_crop = transforms.RandomCrop((img_height_crop, img_width_crop))
    vertical_flip = transforms.RandomVerticalFlip(0.5)

    tensorfy = transforms.ToTensor

    normalize = transforms.Normalize(mean, std)

    test_transforms = transforms.Compose(
        [resize, center_crop, tensorfy, normalize]
    )
    train_transforms = transforms.Compose([
        resize, rotate, random_crop,
        vertical_flip, tensorfy, normalize]
    )
    return train_transforms, test_transforms
