import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DREAM_CONFIG = {
    'ratio': 1.8,
    'levels': 4,
    'num_iters': 10,
    'target_shape': 600,
    'lr': 0.09,
    'out_info': [('fc', None)],
    'shift': True,
    'shift_size': 32,
    'loss_type': 'norm',
    'loss_red': 'mean',
    'eps': 10e-8,
    'clamp_type': 'norm',
    'smooth': True,
    'kernel_size': 9,
    'smooth_coef': 0.5,
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,
    'show': True,
    'use_noise': None,
    'noise_shape': (224, 448),
    'figsize': (15, 15),
    'device': 'cuda:0',
    'save_interval': 5,
    'img_path': None,
    'dpi': 200,
    'video_path': None}