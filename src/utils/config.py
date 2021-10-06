import numpy as np
import copy
from torchvision import  models

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BEETLENET_MEAN = np.array([0.8442649, 0.82529384, 0.82333773], dtype=np.float32)
BEETLENET_STD = np.array([0.28980458, 0.32252666, 0.3240354], dtype=np.float32)

BEETLENET_PATH = 'data/beetles/images/'

RESNET50 = models.resnet50(pretrained=True)

DREAM_CONFIG = {
    'model' : RESNET50,
    'ratio': 1.8,
    'levels': 4,
    'num_iters': 10,
    'target_shape': 600,
    'lr': 0.09,
    'out_info': [('fc', None)],
    'shift_size': 32,
    'loss_type': 'norm',
    'loss_red': 'mean',
    'eps': 10e-8,
    'clamp_type': 'standardize',
    'smooth': True,
    'kernel_size': 9,
    'smooth_coef': 0.5,
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,
    'show': True,
    'noise': None,
    'figsize': (15, 15),
    'device': 'cuda',
    'save_interval': 5,
    'output_img_path': None,
    'dpi': 200,
    'norm_type': 'standardize',
    'video_path': None}

def get_new_config(param_dict, input_img_name,
                   root_img=None, root_video=None):
    output_img_name = input_img_name

    config = copy.deepcopy(DREAM_CONFIG)
    for (name, param) in param_dict.items():
        output_img_name += ('_' + name + '_' + str(param))
        config[name] = param

    if root_img is not None:
        config['output_img_path'] = root_img + '/' + output_img_name + '.jpg'
    if root_video is not None:
        config['video_path'] = root_video + '/' + output_img_name + '.gif'
    return config
