import os
import glob
import ssl

import numpy as np
from PIL import Image

import torch
from torchvision.datasets.utils import download_url, extract_archive
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .config import BEETLENET_STD, BEETLENET_MEAN, RNG_SEED

def download_dataset(url='https://sid.erda.dk/share_redirect/heaAFNnmaG/data.zip',
                     zip_name='beetles.zip', folder_name='beetles',
                     force_download=False, root='./data/'):
    ssl._create_default_https_context = ssl._create_unverified_context
    archive = os.path.join(root, zip_name)
    data_folder = os.path.join(root, folder_name)
    if (not os.path.exists(data_folder) or force_download):
        download_url(url, root, zip_name)
        extract_archive(archive, data_folder, False)
    image_folder_dirs = os.walk(data_folder+'/images/')
    next(image_folder_dirs)
    for root, dirs, files in image_folder_dirs:
        if len(files) < 10:
            print(root)
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)
    return data_folder

def image_folder_dims(data_folder, ext = 'jpg', load_path = None, save_path = None):

    if load_path is not None:
        dims = np.load(load_path)

    else:
        dims_list = []

        for filename in glob.iglob(data_folder + '/**/*.' + ext, recursive=True):
            im = Image.open(filename)
            dims_list.append(im.size)

        dims_list = np.array(dims_list)
        avg_dims = np.mean(dims_list, axis=0)[::-1]
        min_dims = np.min(dims_list, axis=0)[::-1]
        max_dims = np.max(dims_list, axis=0)[::-1]
        dims = np.array((avg_dims, min_dims, max_dims))
    
    if save_path is not None:
        np.save(save_path, dims)
        
    return dims

def image_folder_classes(data_folder):
     return len(next(os.walk(data_folder))[1])

def split_dataset(dataset, train_ratio, val_ratio):

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * (len(dataset) - train_size))
    test_size = len(dataset) - (train_size + val_size)
    dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
    train_data, val_data, test_data = random_split(dataset, dataset_sizes.values(),
                                                    generator=torch.manual_seed(RNG_SEED))
    return train_data, val_data, test_data, dataset_sizes

def split_dataset_stratified(dataset, train_ratio, val_ratio):
    dataset_indices = list(range(len(dataset.targets)))
    train_indices, test_indices = train_test_split(dataset_indices, train_size=train_ratio, 
                                                    stratify=dataset.targets, random_state=RNG_SEED)
    new_targets = np.delete(dataset.targets, test_indices, axis=0)
    train_indices, val_indices  = train_test_split(train_indices, train_size=train_ratio, 
                                                    stratify=new_targets, random_state=RNG_SEED)
    train_data  = Subset(dataset, train_indices)
    test_data   = Subset(dataset, test_indices)
    val_data    = Subset(dataset, val_indices)
    dataset_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
    return train_data, val_data, test_data, dataset_sizes



def dataset_stats(data_set, num_workers=0, batch_size=32):

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

        b, _, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels
    mean = fst_moment.cpu().detach().numpy()
    std = torch.sqrt(snd_moment - fst_moment ** 2).cpu().detach().numpy()
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


def standardize_stats(train_data, shape=(224, 448), num_workers=0, 
                        batch_size=32, load_path=None, save_path=None):

    if load_path is not None:
        mean, std = np.load(load_path)

    else:
        resize = transforms.Resize(shape)
        tensorfy = transforms.ToTensor()
        transforms_pre_norm = transforms.Compose([resize, tensorfy])
        train_data_pre_norm = TransformsDataset(train_data, transforms_pre_norm)
        mean, std = dataset_stats(train_data_pre_norm, num_workers, batch_size)

    if save_path is not None:
        np.save(save_path, np.array((mean, std)))

    return mean, std

def default_transform(train_data, val_data, test_data, shape = (224, 448), 
                        mean = BEETLENET_MEAN, std = BEETLENET_STD):
    resize = transforms.Resize(shape)
    tensorfy = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([resize, tensorfy, normalize])
    train_data_T = TransformsDataset(train_data, transform)
    val_data_T = TransformsDataset(val_data, transform)
    test_data_T = TransformsDataset(test_data, transform)

    return train_data_T, val_data_T, test_data_T

def apply_transforms(transform_list, train_data, val_data , test_data):
    default_shape = (224, 448)
    default_mean = BEETLENET_MEAN
    default_std  = BEETLENET_STD

    default_transforms = transforms.Compose([
        transforms.Resize(default_shape),
        transforms.ToTensor(),
        transforms.Normalize(default_mean, default_std)
    ])
    transform = transforms.Compose(transform_list + [default_transforms])

    train_data_T = TransformsDataset(train_data, transform)
    val_data_T = TransformsDataset(val_data, default_transforms)
    test_data_T = TransformsDataset(test_data, default_transforms)

    return train_data_T, val_data_T, test_data_T

def augment_1(start_height, start_width, scale = 0.95, theta = 3, 
                        mean = BEETLENET_MEAN, std = BEETLENET_STD):
    img_height_crop, img_width_crop = int(start_height * scale), int(start_width * scale)
    resize = transforms.Resize((start_height, start_width))
    center_crop = transforms.CenterCrop((img_height_crop, img_width_crop))

    rotate = transforms.RandomRotation((theta, theta))
    random_crop = transforms.RandomCrop((img_height_crop, img_width_crop))
    vertical_flip = transforms.RandomVerticalFlip(0.5)

    tensorfy = transforms.ToTensor()

    normalize = transforms.Normalize(mean, std)

    test_transforms = transforms.Compose(
        [resize, center_crop, tensorfy, normalize]
    )
    train_transforms = transforms.Compose([
        resize, rotate, random_crop,
        vertical_flip, tensorfy, normalize]
    )
    return train_transforms, test_transforms

def get_dataloaders(train_data, val_data, test_data, batch_size = 32, num_workers = 0):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=True, 
                                            generator=torch.manual_seed(RNG_SEED))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                            num_workers=num_workers, 
                                            generator=torch.manual_seed(RNG_SEED))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            num_workers=num_workers, 
                                            generator=torch.manual_seed(RNG_SEED))
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
