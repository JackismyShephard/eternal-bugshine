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
from .visual import add_noise, get_solid_color

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

def list_classes(data_folder):
    dir = os.walk(data_folder)
    next(dir)
    for i, entry in enumerate(dir):
        print('{}: {}'.format(i, entry[0]))

def show_class_name(i, data_folder):
    dir = os.walk(data_folder)
    next(dir)
    dir = list(dir)
    print('{}: {}'.format(i, dir[0][0]))

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

def get_dataloaders(train_data, val_data, test_data, batch_size = 32, num_workers = 0):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            num_workers=num_workers)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

class RandomizeBackground:
    """Replace beetle PIL image background color with a random color."""
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, img: Image):
        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        
        r = torch.rand(1).item()
        g = torch.rand(1).item()
        b = torch.rand(1).item()
        new_bg = get_solid_color([r,g,b], [np_x.shape[0], np_x.shape[1]])
        np_x = np.where(mask == True, new_bg, np_x)
        np_x = (np_x * 255).astype('uint8')
        return Image.fromarray(np_x)

class NotStupidRandomResizedCrop:
    """A RandomResizedCrop reimplementation that does just what we need it to do.
        Crops a section of a PIL image with shape d*img.shape, where
        min_scale/100 <= d <= max_scale/100, at some random coordinate in the image."""
    def __init__(self, min_scale: float = 0.5, max_scale: float = 1):
        self.rng = np.random.default_rng()
        self.min_scale = int(min_scale * 100)
        self.max_scale = int(max_scale * 100)

    def __call__(self, img: Image):
        np_x = np.array(img)
        scale = self.rng.integers(low=self.min_scale, high=self.max_scale) / 100
        (h, w, _) = np_x.shape
        height = int(scale * h)
        width = int(scale * w)
        x_pos = self.rng.random()
        y_pos = self.rng.random()
        y_max = h - height
        x_max = w - width
        left = int(x_pos * x_max)
        top = int(y_pos * y_max)
        img = transforms.functional.crop(img, top, left, height, width)
        img = transforms.functional.resize(img, [h,w])
        return img

class RandomizeBackgroundGraytone:
    """Replace beetle PIL image background color with a random graytone."""
    def __init__(self, cutoff: float, min: float = 0, max: float = 1):
        self.rng = np.random.default_rng()
        self.cutoff = cutoff
        self.min = min
        self.max = max
    def __call__(self, img: Image):
        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        color = self.rng.integers(int(self.min * 255), int(self.max * 255))
        np_x = np.where(mask == True, color, (np_x * 255).astype('uint8'))
        return Image.fromarray(np_x)

class RandomizeBackgroundRGBNoise:
    """Replace beetle PIL image background color with RGB noise."""
    def __init__(self, cutoff: float):
        self.rng = np.random.default_rng()
        self.cutoff = cutoff
        
    def __call__(self, img: Image):
        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        new_bg = self.rng.random(np_x.shape)
        np_x = np.where(mask == True, new_bg, np_x)
        np_x = (np_x * 255).astype('uint8')
        return Image.fromarray(np_x)

#TODO allow holes to be filled with noise or perhaps solid colors?
class CoarseDropout:
    def __init__(self,  min_holes: int = 0, max_holes:int = 10, 
                        min_height: int = 5, max_height: int = 10, 
                        min_width:int = 5, max_width: int = 10):
        self.rng = np.random.default_rng()
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width

    def __call__(self, img: Image):
        np_x = np.array(img)
        (h, w, _) = np_x.shape
        mask = np.ones(np_x.shape)
        holes = self.rng.integers(self.min_holes, self.max_holes)
        for _ in range(holes):
            width = self.rng.integers(self.min_width, self.max_width)
            height = self.rng.integers(self.min_height, self.max_height)
            x = self.rng.integers(0, w)
            y = self.rng.integers(0, h)
            mask[y:y+height,x:x+width,:] = 0
        np_x = (mask * np_x).astype('uint8')
        return Image.fromarray(np_x)