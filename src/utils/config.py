import csv
import numpy as np
import copy
import os
from torchvision import  models

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEETLENET_MEAN = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)

BEETLENET_PATH = 'data/beetles/images/'

RESNET50 = models.resnet50(pretrained=True)

DREAM_CONFIG = {
    'model': RESNET50,
    'out_info': [('fc', None)],
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,

    #'input_img_path': BEETLENET_PATH + 'achenium_humile/_0189_0.jpg',
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

    'show': True,
    'figsize': (15, 15),
    'save_interval': 1,
    'dpi': 200,
    'output_img_path': None,
    'img_overwrite' : False,
    'video_path': None,
    'video_overwrite' : False,
 
    'device': 'cuda'
}


def get_new_config(param_dict):

    config = copy.deepcopy(DREAM_CONFIG)
    for (name, param) in param_dict.items():
        config[name] = param
    return config

def save_config(config, path):
    (root, _) = os.path.splitext(path)
    csv_path = root + '.csv'
    with open(csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for (name, param) in config.items():
            if name == 'model':
                param_str = type(param).__name__
            elif name in ['output_img_path', 'video_path', 'show']:
                continue
            else:
                param_str = str(param)
            csv_writer.writerow([name, param_str])
    return csv_path


def extend_path(path, overwrite=False):
    if overwrite:
        return path
    else:
        (root, ext) = os.path.splitext(path)
        i = 0
        while os.path.exists(root + str(i) + ext):
            i += 1
        return (root + str(i) + ext)
