import typing as t
import torch
import numpy as np
from numpy import typing as npt
import copy
import os
import json
from ..models import get_model, Exposed_model
from .custom_types import *

RNG_SEED = 0x1010101

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEETLENET_MEAN = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)

BEETLENET_PATH = 'data/beetles/images/'

RESNET50 = Exposed_model(get_model('resnet50'), flatten_layer = 'fc')



DREAM_CONFIG: DreamConfig = {
    'model': RESNET50,
    'model_aux_dict': None,
    'out_info': {'fc': None}, #None = whole layer, otherwise specify index as tuple (y,x).
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,

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
    'video_overwrite' : False,
 
    'device': None
}

#TODO construct model config for model toolchain simplification
MODEL_CONFIG = {
    'model_name': None, #used with load_model
    'model_architecture': 'resnet34',
    'dataset_name': None, #should be beetle dataset path by default
    'dataset_transform': None, #just here for now, will probably be hardcoded in fetch_model or train_model
    'dataloader_args': {'batch_size':32},

    'optim_args': {'lr': 0.001, 'eps':0.1},

}

def get_new_config(param_dict, old_config: DreamConfig = DREAM_CONFIG):

    config = copy.deepcopy(old_config)
    for (name, param) in param_dict.items():
        config[name] = param
    return config


def extend_path(path, overwrite=False):
    if overwrite:
        return path
    else:
        (root, ext) = os.path.splitext(path)
        i = 0
        while os.path.exists(root + str(i) + ext):
            i += 1
        return (root + str(i) + ext)


def save_config(config: DreamConfig, path):
    (root, _) = os.path.splitext(path)
    json_path = root + '.json'
    new_config = copy.deepcopy(config)
    for param in ['mean', 'std']:
        new_config[param] = new_config[param].tolist()
    for param in ['output_img_path', 'video_path', 'show']:
        new_config.pop(param)
    new_config['device'] = new_config['device']
    new_config['model_aux_dict'] = new_config['model'].aux_dict
    with open(json_path, 'w') as json_file:
        json.dump(new_config, json_file, indent = 4)
