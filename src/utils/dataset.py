import os
import glob
import ssl
import typing as t

import numpy as np
from PIL import Image
import cv2 as cv
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split

from torchvision.datasets.utils import download_url, extract_archive
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
from numbers import Number
from torchvision.datasets import ImageFolder
from PIL import Image

from .config import BEETLENET_STD, BEETLENET_MEAN, BEETLENET_AVERAGE_SHAPE, RNG_SEED
from .custom_types import *


def download_dataset(url : str ='https://sid.erda.dk/share_redirect/heaAFNnmaG/data.zip',
                     zip_name : str ='beetles.zip', folder_name : str ='beetles',
                     force_download : bool =False, root : str='./data/', min_examples : int = 10) -> str:
    ssl._create_default_https_context = ssl._create_unverified_context
    archive = os.path.join(root, zip_name)
    data_folder = os.path.join(root, folder_name)
    if (not os.path.exists(data_folder) or force_download):
        download_url(url, root, zip_name)
        extract_archive(archive, data_folder, False)
    image_folder_dirs = os.walk(data_folder+'/images/')
    next(image_folder_dirs)
    for root, _, files in image_folder_dirs:
        if len(files) < min_examples:
            print(root)
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)
    return data_folder

def image_folder_dims(data_folder : str, ext :IMG_EXT = '.jpg', 
                        load_path: t.Optional[str] = None, 
                        save_path: t.Optional[str] = None) -> npt.NDArray[np.float64]:
    if load_path is not None:
        dims = np.load(load_path)

    else:
        dims_list = []

        for filename in glob.iglob(data_folder + '/**/*' + ext, recursive=True):
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

def image_folder_classes(data_folder : str) -> int:
    '''Get number of different classes in image folder.'''
    return len(next(os.walk(data_folder))[1])

def list_classes(dataset_config: DatasetConfig, h_split : int = 6) -> None:
    classes = list(os.listdir(dataset_config['image_folder_path']))
    classes.sort()
    classes = np.array(classes).reshape(-1,1)
    class_nr = np.arange(classes.shape[0]).astype(str).reshape(-1,1)

    table = np.hstack((class_nr, classes))

    # reorganise to use the horizontal space better
    even_div = table.shape[0]//h_split + 1
    remaining = even_div * 6 - table.shape[0]
    placeholder = np.full((remaining, 2), ' ')
    table = np.vstack((table,placeholder))

    new_table = np.full((even_div,h_split*2), ' '*64)
    for i in range(h_split):
        new_table[:,i*2:(i+1)*2] = table[i*even_div:(i+1)*even_div,:]

    pdf = pd.DataFrame(new_table, columns=['Class', 'Species name']*h_split)
    
    with pd.option_context('display.max_rows',None):
        display(pdf.style.hide_index())

def show_class_name(i : int, dataset_config: DatasetConfig) -> None: 
    dir = os.walk(dataset_config['image_folder_path'])
    next(dir)
    dir = list(dir)
    dir.sort()
    print('class {}: {}'.format(i, dir[i][0]))

def show_class_name2(i : int, dataset_config: DatasetConfig) -> None:
    '''similar to get_class_example_image, but class name instead of full paths are printed'''
    dir = os.walk(dataset_config['image_folder_path'])
    _, classes, _ = next(dir)
    classes.sort()
    print('class {}: {}'.format(i, classes[i]))

def get_class_example_image(i : int, dataset_config: DatasetConfig) -> npt.NDArray[np.uint8]:
    dir = os.walk(dataset_config['image_folder_path'])
    next(dir)
    dir = list(dir)
    dir.sort()
    path_prefix = dir[i][0]
    print('Class path/name: {}'.format(path_prefix))
    dir = os.walk(dir[i][0])
    dir = list(dir)
    path_suffix = dir[0][2][0]
    path = os.path.join(path_prefix, path_suffix)
    return cv.imread(path)[:, :, ::-1]


def get_class_example_image2(i : int, dataset_config: DatasetConfig) -> npt.NDArray[np.uint8]:
    '''similar to get_class_example_image, but class name instead of full paths are printed'''
    dir = os.walk(dataset_config['image_folder_path'])
    _, classes, _ = next(dir)
    classes.sort()
    print('Class: {}'.format(classes[i]))
    dir = list(dir)
    dir.sort()
    path_prefix = dir[i][0]
    dir = os.walk(dir[i][0])
    dir = list(dir)
    path_suffix = dir[0][2][0]
    path = os.path.join(path_prefix, path_suffix)
    return cv.imread(path)[:, :, ::-1]


def split_dataset_stratified(dataset : ImageFolder , train_ratio: t.Union[int, float] = 0.8, 
                            val_ratio: t.Union[int, float] = 0.5) -> t.Tuple[Subset, Subset, Subset, t.Dict[str, int]]:
    '''Performs a stratified split of an Image Folder dataset into training, validation and test sets.
       train_ratio indicates the relative ratio of training examples with respect to the original dataset.
       val_ratio indicates the relative ratio of validation examples with respect to the dataset minus the training
       examples
    '''
    dataset_indices = list(range(len(dataset.targets)))
    train_indices, test_indices = train_test_split(dataset_indices, train_size=train_ratio, 
                                                    stratify=dataset.targets, random_state=RNG_SEED)
    new_targets = np.delete(dataset.targets, train_indices, axis=0)
    val_indices, test_indices  = train_test_split(test_indices, train_size=val_ratio, 
                                                    stratify=new_targets, random_state=RNG_SEED)
    train_data  = Subset(dataset, train_indices)
    test_data   = Subset(dataset, test_indices)
    val_data    = Subset(dataset, val_indices)
    dataset_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
    return train_data, val_data, test_data, dataset_sizes


class TransformsDataset(Dataset):
    '''Transforms a Dataset subset by a given transformation'''
    def __init__(self, subset: Subset, transform: t.Optional[t.Union[torch.nn.Module, ToTensor, Compose]] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index : int) -> t.Any:
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


def standardize_stats(train_data : Subset, shape : t.Union[int, t.Sequence[int]]=(224, 448), num_workers: int=0,
                      batch_size: int = 32, load_path:  t.Optional[str] = None, 
                      save_path:  t.Optional[str] = None) -> t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''Retrieves the mean and std of an dataset subset after resizing to given dimensions and tensorfying'''
    if load_path is not None:
        mean, std = np.load(load_path)

    else:
        resize = Resize(shape)
        tensorfy = ToTensor()
        transforms_pre_norm = Compose([resize, tensorfy])
        train_data_pre_norm = TransformsDataset(
            train_data, transforms_pre_norm)
        mean, std = dataset_stats(train_data_pre_norm, num_workers, batch_size)

    if save_path is not None:
        np.save(save_path, np.array((mean, std)))

    return mean, std


def dataset_stats(data_set: Dataset[torch.Tensor], num_workers: int = 0, 
                    batch_size: int = 32) -> t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    '''Computes the mean and std of a dataset of tensors.'''
    loader = DataLoader(
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



def apply_transforms(transform_list : t.List[t.Union[ToTensor, torch.nn.Module]], 
                        train_data: Subset, val_data : Subset , test_data :  Subset, 
                        default_shape: t.Union[int, t.Sequence[int]] =BEETLENET_AVERAGE_SHAPE,
                        default_mean: npt.NDArray[np.float32] = BEETLENET_MEAN, 
                        default_std: npt.NDArray[np.float32] = BEETLENET_STD) -> t.Tuple[Dataset[torch.Tensor], Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    
    default_transforms = Compose([Resize(default_shape), ToTensor(), Normalize(default_mean, default_std)
    ])
    transform = Compose(transform_list)

    train_data_T = TransformsDataset(train_data, transform)
    val_data_T = TransformsDataset(val_data, default_transforms)
    test_data_T = TransformsDataset(test_data, default_transforms)

    return train_data_T, val_data_T, test_data_T

def get_dataloaders(train_data : Dataset, val_data : Dataset, test_data : Dataset, 
                    batch_size : int = 32, num_workers : int = 0) -> t.Dict[str, DataLoader]:
    train_loader = DataLoader(train_data, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                                            num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                                            num_workers=num_workers)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def dataset_to_dataloaders(dataset_config: DatasetConfig) -> t.Tuple[t.Dict[str, DataLoader], t.Dict[str, int]]:
    """Create dataloaders from dataset images.\n
       Returns: training_dataset, validation_dataset, testing_dataset"""
    #FIXME currently assumes all given paths are split with '/'
    dataset = ImageFolder(dataset_config['image_folder_path'])
    training_ratio = dataset_config['training_data_ratio']
    validation_ratio = dataset_config['validation_data_ratio']
    train_data, val_data, test_data, dataset_sizes = split_dataset_stratified(
        dataset, training_ratio, validation_ratio)
    print('dataset sizes: {}'.format(dataset_sizes))
    transforms = dataset_config['data_augmentations']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']
    train_dataset, val_dataset, test_dataset = apply_transforms(
        transforms, train_data, val_data, test_data, 
        dataset_config['average_image_shape'],  dataset_config['mean'], dataset_config['std'])
    dataloaders = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers)
    return dataloaders, dataset_sizes
