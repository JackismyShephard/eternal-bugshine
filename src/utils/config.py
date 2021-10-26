

import multiprocessing as mp
from pathlib import Path
import os
import getpass

import copy
import json
import typing as t

import numpy as np
from numpy import typing as npt

from .custom_types import *
from .transforms import *
from torch.utils.data import DataLoader

RNG_SEED = 0x1010101

IMAGENET_MEAN : npt.NDArray[np.float32] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: npt.NDArray[np.float32] = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEETLENET_MEAN: npt.NDArray[np.float32] = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD: npt.NDArray[np.float32] = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)
BEETLENET_AVERAGE_SHAPE: t.Tuple[float, float]= (224, 448)

BEETLENET_PATH : str = './data/beetles/images/' 
BEETLENET_NUM_CLASSES : int = len(next(os.walk(BEETLENET_PATH))[1])

DEVICE : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_NUM_WORKERS : int = mp.cpu_count()//2


DEFAULT_OUTPUT_PATH : str = './output/' + getpass.getuser() + '/'
Path(DEFAULT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
DEFAULT_IMG_PATH : str = DEFAULT_OUTPUT_PATH + 'figures/'
Path(DEFAULT_OUTPUT_PATH + 'figures').mkdir(parents=True, exist_ok=True)
DEFAULT_VIDEO_PATH: str = DEFAULT_OUTPUT_PATH + 'videos/'
Path(DEFAULT_OUTPUT_PATH + 'videos').mkdir(parents=True, exist_ok=True)

Path(DEFAULT_OUTPUT_PATH + 'models').mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH : str = 'models/'
DEFAULT_METRICS_PATH : str = 'models/'
Path('models').mkdir(parents=True, exist_ok=True)

DREAM_CONFIG: DreamConfig = {
    'target_dict': {'fc': None}, #None = whole layer, otherwise specify index as tuple (y,x). 
    'mean': BEETLENET_MEAN,
    'std': BEETLENET_STD,
    'input_img_path': None,
    'target_shape': 600,
    
    'noise': None,
     'noise_scale' : 1.0,
    'correlation' : None,
    'correlation_std' : 1.0,
    'ratio': 1.8,
    'levels': 4,
    'gauss_filter' : None,
    'shift_size': 32,
    'num_iters': 10,
    'lr': 0.05,
    'loss_type': 'norm',
    'loss_red': 'mean',
    'norm_type': 'standardize',
    'eps': 10e-8,
    'smooth': True, 
    'kernel_size': 9,
    'smooth_const': 0.5,
    'smooth_factors': [1, 2, 5],
    'clamp_type': 'standardize',
    
    'show': False,
    'figsize': (15, 15),
    'save_interval': 1,
    'dpi': 200,

    'output_img_path': None,
    'output_img_ext': '.jpg',
    'img_overwrite' : False,
    'video_path': None,
    'video_ext': '.mp4',
    'video_overwrite' : False,
    'add_path_info': True
    
}

BEETLE_DATASET: DatasetConfig = {
    'image_folder_path':    BEETLENET_PATH,
    'num_classes':          BEETLENET_NUM_CLASSES,
    'batch_size':           32,
    'num_workers':          DEFAULT_NUM_WORKERS,
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

RESNET34_FULL_WHITE: ModelConfig = {
    'model_name':           'resnet34_fullytrained_white',
    'model_architecture':   'resnet34',
    'pretrained':           False,
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
    'early_stopping_args':  {'min_epochs': 400, 'patience': 5, 'min_delta': 0},
    'train_info':           {'num_epochs': 400, 'best_model_epochs': 0, 
                             'lr_decay': 0.995, 'stopped_early': False,
                             'test_acc':   0, 'best_model_val_acc' : 0.0, 
                             'best_model_val_loss' : float("inf")},
    'metrics_path' : DEFAULT_METRICS_PATH,
    'model_path': DEFAULT_MODEL_PATH
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
    'save_suffix':       'comparison',
    'save_ext':          '.png',
    'param_linestyle':      'solid',
    'average_linestyle':    'dashed',
    'param_alpha':          0.5,
    'rolling_avg_window':   50,
    'rolling_avg_label':    'average',
    'show_rolling_avg':     True
}


def get_new_config(param_dict: t.Dict, old_config: t.Union[ETERNAL_CONFIG, t.Dict],
                   new_keys: bool = False) -> t.Union[ETERNAL_CONFIG, t.Dict]:

    if not(new_keys) and not(param_dict.keys() <= old_config.keys()):
        raise RuntimeError('param_dict keys must be subset of old_config keys')

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

RESNET34_FULL_COARSE: ModelConfig = {
    'model_name':           'resnet34_fullytrained_coarse_dropout',
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
jens_params2 = {
    'data_augmentations':   [
                                RandomVerticalFlip(),
                                RandomRotation((-3,3), fill=255),
                                NotStupidRandomResizedCrop(min_scale=0.95, max_scale=1),
                                RandomizeBackground(cutoff=0.95),
                                CoarseDropout(0, 15, 15, 40, 15, 40, 'black'),
                                Resize(BEETLENET_AVERAGE_SHAPE),
                                ToTensor(),
                                Normalize(BEETLENET_MEAN, BEETLENET_STD)
                            ]
}
JENS_DATASET = get_new_config(jens_params2, BEETLE_DATASET)

JACKI_PARAM = {
    'data_augmentations':   [
        CoarseDropout(min_holes =5, max_holes =15,
                      min_height =20, max_height =40,
                      min_width =20, max_width =40,
                      fill_type = None),
        Resize(BEETLENET_AVERAGE_SHAPE),
        ToTensor(),
        Normalize(BEETLENET_MEAN, BEETLENET_STD)
    ]
}

JACKISET = get_new_config(JACKI_PARAM, BEETLE_DATASET)


def add_info_to_path(path: str, info: t.Optional[str], new_ext: t.Optional[IMG_EXT] = None, 
                        overwrite: bool = False) -> str:
    '''Adds new information to path string. Main information to be appended is given by info.
       If new_ext is not None then any current file extension is replaced by it. If overwrite is False
       (Default) then an integer is suffixed to the resulting path string (before its extension) in such
       a way that the resulting string identifies a new file in the system'''

    (root, old_ext) = os.path.splitext(path)
    if new_ext is not None:
        old_ext = new_ext
    if info is not None:
        root = root + info
    if overwrite:
        return root + old_ext
    else:
        i = 0
        while os.path.exists(root + '_' + str(i) + old_ext):
            i += 1
        return (root  + '_' + str(i) + old_ext)


def save(path :str, model_config: ModelConfig, dataset_config: DatasetConfig, 
         training_config: TrainingConfig, model_state: t.OrderedDict[str, torch.Tensor] = None,
         optim_state: dict = None, dataloaders: t.Dict[str, DataLoader]=None, 
        train_metrics: t.Optional[npt.NDArray] = None, 
        dream_config: t.Optional[DreamConfig] = None) -> None:
    (root, _) = os.path.splitext(path)
    json_path = root + '_aux_dict.json'

    if model_state is not None:
        torch.save(model_state, path + '_parameters.pt')
    if optim_state is not None:
        torch.save(optim_state, path + '_optim.pt')
    if dataloaders is not None:
        torch.save(dataloaders, path + '_dataloaders.pt')
    if train_metrics is not None:
        np.save(path + '_train_metrics.npy', train_metrics)

    config = {'model_info': model_config, 'dataset_info': dataset_config, 
                  'train_info': training_config, 'dream_info':dream_config}

    new_config = copy.deepcopy(config)

    #convert troublesome dict entries:
    new_config['model_info']['device'] = str(new_config['model_info']['device'])

    new_config['dataset_info']['data_augmentations'] = [str(aug) for aug in new_config['dataset_info']['data_augmentations']]
    new_config['dataset_info']['mean'] = new_config['dataset_info']['mean'].tolist()
    new_config['dataset_info']['std'] = new_config['dataset_info']['std'].tolist()
  
    new_config['train_info']['optim'] = str(new_config['train_info']['optim'])
    new_config['train_info']['criterion'] = str(new_config['train_info']['criterion'])
    new_config['train_info']['early_stopping'] = str(new_config['train_info']['early_stopping'])
    new_config['train_info']['scheduler'] = str(new_config['train_info']['scheduler'])

    if dream_config is not None:
        new_config['dream_info']['mean'] = new_config['dream_info']['mean'].tolist()
        new_config['dream_info']['std'] = new_config['dream_info']['std'].tolist()
    else:
        new_config.pop('dream_info')
    
    with open(json_path, 'w') as json_file:
        json.dump(new_config, json_file, indent = 4)
