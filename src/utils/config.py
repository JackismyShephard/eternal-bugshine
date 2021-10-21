import typing as t
import torchvision.transforms
from .transforms import *
import multiprocessing as mp
import numpy as np
from pathlib import Path
from numpy import typing as npt
import copy
import os
import json
import getpass
from .custom_types import *

RNG_SEED = 0x1010101

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEETLENET_MEAN = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)
BEETLENET_AVERAGE_SHAPE = (224, 448)

BEETLENET_PATH = './data/beetles/images/' 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEFAULT_OUTPUT_PATH = './output/' + getpass.getuser() + '/'
Path(DEFAULT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(DEFAULT_OUTPUT_PATH + 'figures').mkdir(parents=True, exist_ok=True)
Path(DEFAULT_OUTPUT_PATH + 'videos').mkdir(parents=True, exist_ok=True)
def mkdir_user():
    Path(DEFAULT_OUTPUT_PATH).mkdir(exist_ok=True, parents=True)
    return DEFAULT_OUTPUT_PATH



DEFAULT_MODEL_PATH = './models/'
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_PATH + 'figures/'

DREAM_CONFIG: DreamConfig = {
    'out_info': {'fc': None}, #None = whole layer, otherwise specify index as tuple (y,x). 
    'mean': BEETLENET_MEAN,
    'std': BEETLENET_STD,
    'input_img_path': None,
    'target_shape': 600,
    'noise': None,
    'ratio': 1.8,
    'levels': 4,
    'shift_size': 32,
    'num_iters': 10,
    'lr': 0.09,
    'loss_type': 'norm',
    'loss_red': 'mean',
    'norm_type': 'standardize',
    'eps': 10e-8,
    'smooth': True, 
    'kernel_size': 9,
    'smooth_coef': 0.5,
    'clamp_type': 'standardize',
    
    'show': False,
    'figsize': (15, 15),
    'save_interval': 1,
    'dpi': 200,
    'output_img_path': None,
    'img_overwrite' : False,
    'video_path': None,
    'video_overwrite' : False
}

BEETLE_DATASET: DatasetConfig = {
    'image_folder_path':    BEETLENET_PATH,
    'num_classes':          len(next(os.walk(BEETLENET_PATH))[1]),
    'batch_size':           32,
    'num_workers':          (0 if os.name == "nt" else mp.cpu_count()//2),
    'rng_seed':             RNG_SEED,
    'average_image_shape':  BEETLENET_AVERAGE_SHAPE,
    'mean':                 BEETLENET_MEAN,
    'std':                  BEETLENET_STD,
    'training_data_ratio':  0.8,
    'validation_data_ratio':0.5,
    'data_augmentations':   [
                                RandomVerticalFlip(),
                                RandomRotation((-3,3), fill=255),
                                NotStupidRandomResizedCrop(min_scale=0.95, max_scale=1),
                                RandomizeBackground(cutoff=0.95),
                                Resize(BEETLENET_AVERAGE_SHAPE),
                                ToTensor(),
                                Normalize(BEETLENET_MEAN, BEETLENET_STD)
                            ]
}


RESNET34_FULL_RGB: ModelConfig = {
    'model_name':           'resnet34_fullytrained_rgb',
    'model_architecture':   'resnet34',
    'pretrained':           False,
    'device':               DEVICE
}

RESNET50_FULL: ModelConfig = {
    'model_name':           'resnet50_fullytrained',
    'model_architecture':   'resnet50',
    'pretrained':           False,
    'device':               DEVICE
}

RESNET50_TRANSFER: ModelConfig = {
    'model_name':           'resnet50_transferlearned',
    'model_architecture':   'resnet50',
    'pretrained':           True,
    'device':               DEVICE
}

RESNET34_FULL: ModelConfig = {
    'model_name':           'resnet34_fullytrained',
    'model_architecture':   'resnet34',
    'pretrained':           True,
    'device':               DEVICE
}

RESNET34_TRANSFER: ModelConfig = {
    'model_name':           'resnet34_transferlearned',
    'model_architecture':   'resnet34',
    'pretrained':           True,
    'device':               DEVICE
}

RESNET18_FULL: ModelConfig = {
    'model_name':           'resnet18_fullytrained',
    'model_architecture':   'resnet18',
    'pretrained':           False,
    'device':               DEVICE
}

RESNET18_TRANSFER: ModelConfig = {
    'model_name':           'resnet18_transferlearned',
    'model_architecture':   'resnet18',
    'pretrained':           True,
    'device':               DEVICE
}

RESNET18_TEST: ModelConfig = {
    'model_name':           'resnet18_test',
    'model_architecture':   'resnet18',
    'pretrained':           False,
    'device':               DEVICE
}

DEFAULT_TRAINING: TrainingConfig = {
    
    'optim':                None,
    'optim_args':           {'lr': 0.001, 'eps': 0.1},
    'criterion':            None,
    'scheduler':            None,
    'early_stopping':       None,
    'early_stopping_args':  {'min_epochs': 200, 'patience': 5, 'min_delta': 0},
    'train_info':           {'num_epochs': 400, 'trained_epochs': 0, 
                             'lr_decay': 0.995, 'stopped_early': False,
                             'test_acc':   0},
}  

DEFAULT_PLOTTING: PlotConfig = {
    'size_h':               7,
    'size_w':               14,
    'fig_column':           2,
    'fig_row':              1,
    'show_title':           True,
    'titles':               ['Loss', 'Accuracy'],
    'use_title_label':      True,
    'label':                ['Training', 'Validation'],
    'y_label':              None,
    'x_label':              'epoch',
    'show_grid':            True,
    'save_dpi':             200,
    'save_figure':          True,
    'save_subfigures':      True,
    'save_padding':         0,
    'save_extension':       'comparison',
    'save_copy_png':        True,
    'param_linestyle':      'solid',
    'average_linestyle':    'dashed',
    'param_alpha':          0.5,
    'rolling_avg_window':   50,
    'rolling_avg_label':    'average',
    'show_rolling_avg':     True
}

def get_new_config(param_dict, old_config: t.Union[DreamConfig, DatasetConfig, None] = None):
    
    if old_config is None:
        raise TypeError('Config must not be None. See src/utils/config.py for default configs')
    config = copy.deepcopy(old_config)
    for (name, param) in param_dict.items():
        config[name] = param
    return config


mathias_params = {
    'data_augmentations':   [
                                RandomVerticalFlip(),
                                RandomRotation((-3,3), fill=255),
                                NotStupidRandomResizedCrop(min_scale=0.95, max_scale=1),
                                RandomizeBackgroundGraytone(cutoff=0.95, min= 0.2, max=1.0, ),
                                Resize(BEETLENET_AVERAGE_SHAPE),
                                ToTensor(),
                                Normalize(BEETLENET_MEAN, BEETLENET_STD)
                            ]
}

MATHIAS_DATASET = get_new_config(mathias_params, BEETLE_DATASET)

RESNET34_FULL_GRAY: ModelConfig = {
    'model_name':           'resnet34_fullytrained_gray',
    'model_architecture':   'resnet34',
    'pretrained':           False,
    'device':               DEVICE
}

jens_params = {
    'data_augmentations':   [
                                RandomVerticalFlip(),
                                RandomRotation((-3,3), fill=255),
                                NotStupidRandomResizedCrop(min_scale=0.95, max_scale=1),
                                RandomizeBackgroundRGBNoise(cutoff=0.95),
                                Resize(BEETLENET_AVERAGE_SHAPE),
                                ToTensor(),
                                Normalize(BEETLENET_MEAN, BEETLENET_STD)
                            ]
}

JENS_DATASET = get_new_config(jens_params, BEETLE_DATASET)


def extend_path(path, overwrite=False):
    if overwrite:
        return path
    else:
        (root, ext) = os.path.splitext(path)
        i = 0
        while os.path.exists(root + str(i) + ext):
            i += 1
        return (root + str(i) + ext)


def save(path, model_config: ModelConfig, 
                        dataset_config: DatasetConfig, training_config: TrainingConfig,
                        model = None, optim = None, dataloaders = None, train_metrics = None, dream_config: t.Optional[DreamConfig] = None):
    (root, _) = os.path.splitext(path)
    json_path = root + '_aux_dict.json'

    if model is not None:
        torch.save(model.state_dict(), path + '_parameters.pt')
    if optim is not None:
        torch.save(optim.state_dict(), path + '_optim.pt')
    if dataloaders is not None:
        torch.save(dataloaders, path + '_dataloaders.pt')
    if train_metrics is not None:
        np.save(path + '_train_metrics.npy', train_metrics)

    new_config = {}

    new_config.update(model_info=model_config)
    new_config.update(dataset_info=dataset_config)
    new_config.update(train_info=training_config)
    new_config.update(dream_info=dream_config)

    new_config = copy.deepcopy(new_config)

    #convert troublesome dict entries:
    new_config['dataset_info']['data_augmentations'] = [str(aug) for aug in new_config['dataset_info']['data_augmentations']]
    new_config['model_info']['device'] = str(new_config['model_info']['device'])
    new_config['dataset_info']['mean'] = new_config['dataset_info']['mean'].tolist()
    new_config['dataset_info']['std'] = new_config['dataset_info']['std'].tolist()
    if dream_config is not None:
        new_config['dream_info']['mean'] = new_config['dream_info']['mean'].tolist()
        new_config['dream_info']['std'] = new_config['dream_info']['std'].tolist()

    new_config['train_info']['optim'] = str(new_config['train_info']['optim'])
    new_config['train_info']['criterion'] = str(new_config['train_info']['criterion'])
    new_config['train_info']['early_stopping'] = str(new_config['train_info']['early_stopping'])
    new_config['train_info']['scheduler'] = str(new_config['train_info']['scheduler'])
    
    with open(json_path, 'w') as json_file:
        json.dump(new_config, json_file, indent = 4)
