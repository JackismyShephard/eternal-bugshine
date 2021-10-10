import collections

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
from PIL import Image
from skimage import filters as skfilt
from skimage.util import random_noise

from .config import BEETLENET_MEAN, BEETLENET_STD


def multiplot(systems, x_axis, y_axis, labels, save_path=None,
              title=None, dpi=200):
    plt.figure()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    for i in range(len(systems)):
        plt.plot(systems[i, 0], systems[i, 1], label=labels[i])
        plt.title(title, pad=20)
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path,  bbox_inches='tight',
                    facecolor='w', dpi=dpi)
    plt.show()
    plt.close()

def reshape_image(img, shape):
    if isinstance(shape, int):
        current_height, current_width = img.shape[:2]
        new_height = int(current_height * (shape / current_width))
        new_img = cv.resize(img, (shape, new_height),
                            interpolation=cv.INTER_CUBIC)

    elif isinstance(shape, collections.abc.Sequence):
        new_img = cv.resize(
            img, (shape[1], shape[0]), interpolation=cv.INTER_CUBIC)
            
    return new_img

def get_noise_image(type, shape):
    if type == 'uniform':
        (h, w) = shape
        img = np.random.uniform(size=(h, w, 3)).astype(np.float32)
    elif type == 'correlated_uniform':
        (h, w) = shape
        img = np.random.uniform(size=(h, w, 3)).astype(np.float32)
        img = skfilt.gaussian(img, mode='reflect', multichannel=True)
        return img
    elif type == 'correlated_gaussian':
        (h, w) = shape
        img = np.random.normal(size=(h, w, 3)).astype(np.float32)
        img = skfilt.gaussian(img, mode='reflect', multichannel=True)
        return img
    else:
        (h, w) = shape
        img = np.random.normal(size=(h, w, 3)).astype(np.float32)
    return img

def get_solid_color(color, shape):
    _color = color
    if color == 'white':
        _color = [1., 1., 1.]
    if color == 'black':
        _color = [0.,0.,0.]
    if color == 'red':
        _color = [1.,0.,0.]
    if color == 'green':
        _color = [0.,1.,0.]
    if color == 'blue':
        _color = [0.,0.,1.]
    (h, w) = shape
    img = np.zeros(shape=(h,w,3), dtype=np.float32)
    for i in range(3):
        img[:,:,i] = _color[i]
    return img
def apply_noise(type, img):
    return random_noise(img, type)
    

def preprocess_image(img, mean=BEETLENET_MEAN, std=BEETLENET_STD, 
                        range = 255.0):
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= range  # get to [0, 1] range
    img = (img - mean) / std
    return img

def postprocess_image(img, mean=BEETLENET_MEAN, std=BEETLENET_STD):
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    return img


def image_to_tensor(img, device='cuda', requires_grad=False):
    tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    tensor.requires_grad = requires_grad
    return tensor

def tensor_to_image(tensor):
    tensor = tensor.to('cpu').detach().squeeze(0)
    img = tensor.numpy().transpose((1, 2, 0))
    return img


def random_shift(tensor, h_shift, w_shift, undo=False, requires_grad = True):
    if undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = requires_grad
        return rolled


def show_img(img,title=None, save_path=None, dpi=200, figsize=(7, 7), show_axis='on',close = False):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis(show_axis)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',
                    dpi=dpi, facecolor='w')
    if close:
        plt.close()
    plt.pause(0.001)  # pause a bit so that plots are updated

def save_img(img, path):
    cv.imwrite(path, img[:, :, ::-1])

def make_video(images, shape, path):
    imgs = [Image.fromarray(reshape_image(img, shape)) for img in images]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0)
