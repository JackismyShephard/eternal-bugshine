

import numbers
import math

import numpy as np
import numpy.fft as fft

import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


from .utils.config import add_info_to_path, save, Config
from .utils.custom_types import *
from .utils.visual import reshape_image, get_noise_image, tensor_to_image, postprocess_image, image_to_tensor, random_shift, save_img, Rendering, Rendering_stats, save_video
from .utils.deepdream_utility import *


def dream_process(model : torch.nn.Module, config : Config, 
                    img : t.Optional[npt.NDArray[np.float32]] = None, render=None) -> t.List[npt.NDArray[np.uint8]]:
    input_img = get_start_image(config, img)
    output_images = dreamspace(input_img, model, config = config, render=render)
    save_output(output_images, config)
    
    return output_images

def dreamspace(img : npt.NDArray[np.float32], model : torch.nn.Module, 
                    config : Config, render=None) -> t.List[npt.NDArray[np.uint8]]:
    output_images = []

    iters = get_num_iters(config.dream)

    if config.dream['show'] == True and render is None:
            render = Rendering(config.dream['target_shape'])
    image_scale = Image_setup.setup(config, img)

    scaled_tensor = image_scale.get_first_level(img)

    for level in range((config.dream['levels'] - config.dream['end_level'])):
        for i in range(iters[level]):
            h_shift, w_shift = np.random.randint(-config.dream['shift_size'], config.dream['shift_size'] + 1, 2)
            shifted_tensor = random_shift(scaled_tensor, h_shift, w_shift, requires_grad = True)
            dreamt_tensor = dream_ascent(shifted_tensor, model, level, i, iters[level],  config=config, render=render)
            deshifted_tensor = random_shift(dreamt_tensor, h_shift,
                                             w_shift, undo=True, requires_grad = True)

            img = tensor_to_image(deshifted_tensor)
            
            output_image = postprocess_image(img, config.mean ,config.std)
            if config.dream['show'] == True and i % config.dream['display_interval'] == 0:
                render.update(output_image)

            if (i % config.dream['save_interval']) == 0:
                output_images.append(output_image)
            scaled_tensor = deshifted_tensor
        scaled_tensor = image_scale.get_level(img, level)
    return output_images

def calculate_loss(tensor : torch.Tensor, loss_type : str = ""):
    if loss_type == 'norm':
        loss_part = torch.linalg.norm(tensor)
    elif loss_type == 'mean':
        loss_part = torch.mean(tensor)
    else:
        MSE = torch.nn.MSELoss(reduction='mean')
        zeros = torch.zeros_like(tensor)
        loss_part = MSE(tensor, zeros)
    return loss_part

def dream_ascent(tensor : torch.Tensor, model : torch.nn.Module, level : int,  iter : int, num_iters : int,
                config : Config, render=None) -> torch.Tensor:
      
 
    ## get activations
    x, (target_acts, remaining_acts) = model(tensor, config.dream['target_dict'], config.dream['penalty'])
    losses_target = []
    losses_remaining = []

    ### calculate loss on target layers
    for target_activation in target_acts:
        losses_target.append(calculate_loss(torch.stack(target_activation), config.dream['loss_type']))

    if config.dream['loss_red'] == 'mean':
        loss_target = torch.mean(torch.stack(losses_target))
    else:
        loss_target = torch.sum(torch.stack(losses_target))


    if config.dream['penalty']:
        for remaining_activation in remaining_acts:
            if config.dream['penalty_function'] == 'relu':
                remaining_f = F.relu(torch.stack(remaining_activation))
            else:
                remaining_f = remaining_activation

            losses_remaining.append(calculate_loss(remaining_f, config.dream['penalty_loss_type']))

        if config.dream['penalty_red'] == 'mean':
            loss_remaining = torch.mean(torch.stack(losses_remaining))
        else:
            loss_remaining = torch.sum(torch.stack(losses_remaining))

        loss = loss_target - loss_remaining
    else:
        loss = loss_target

    if config.dream['show_stats']:
        render.data = x.cpu().detach().numpy()[0]
        render.loss_target = loss_target.cpu().detach().numpy()
        if config.dream['penalty']:
            render.loss_penalty = loss_remaining.cpu().detach().numpy()

    # do backpropagation and get gradient

    loss.backward()
    grad = tensor.grad.data
    ### gaussian smoothing
    sigma = ((iter + 1) / num_iters) * 2.0 + config.dream['smooth_const']
    smooth_grad = gradient_smoothing(grad, config.dream['kernel_size'], sigma, config.dream['smooth_factors'])
    #smooth_grad = CascadeGaussianSmoothing(kernel_size=dream_config['kernel_size'], sigma=sigma,device=device)(grad)
    ### normalization of gradient
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    if config.dream['norm_type'] == 'standardize':
        smooth_grad = smooth_grad - g_mean
        smooth_grad = smooth_grad / g_std
    else:
        smooth_grad /= torch.abs(g_mean + config.dream['eps'])


    ### gradient update ####
    lr = get_lr(level, config.dream)

    tensor.data += lr * smooth_grad
    tensor.grad.data.zero_()
    ### clamp gradient to avoid it diverging. vanishing/exploding gradient phenomenon?
    if config.dream['clamp_type'] == 'standardize':
        image_min = torch.tensor(
            (-config.mean / config.std).reshape(1, -1, 1, 1)).to(config.device)
        image_max = torch.tensor(
            ((1 - config.mean) / config.std).reshape(1, -1, 1, 1)).to(config.device)
    elif config.dream['clamp_type'] == 'unit':
        image_min, image_max = torch.tensor(0), torch.tensor(1)
    else:
        image_min, image_max = torch.tensor(-1), torch.tensor(1)
    tensor.data = torch.clip(tensor, image_min, image_max)
    return tensor


def gradient_smoothing(tensor: torch.Tensor, kernel_size: t.Union[int, t.List[int]], 
                  sigma: t.Union[t.List[float], float, None] = None, 
                  coefs: t.List[float] = [1, 2, 5]) -> torch.Tensor:

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(sigma, float):
        sigma = [sigma, sigma]
    blurred_imgs = []
    for coef in coefs:
        if sigma is None:
            new_sigma = sigma
        else:
            new_sigma = [sigma[0] * coef, sigma[1] * coef]
        blurred_imgs.append(TF.gaussian_blur(tensor, kernel_size, new_sigma))
    return torch.stack(blurred_imgs).mean(dim = 0)

def get_num_iters(dream_config : DreamConfig):
    if dream_config['iteration_mode'] == 'ratio':
        iter_exp = np.arange(dream_config['levels'])
        start_iter, iter_ratio = dream_config['num_iters']
        return ((iter_ratio ** iter_exp) * start_iter).astype(int)

    elif dream_config['iteration_mode'] == 'custom':
        return dream_config['num_iters']
    else:
        return [dream_config['num_iters']] * dream_config['levels']

def get_lr(level : int, dream_config : DreamConfig):
    if dream_config['lr_mode'] == 'custom':
        return dream_config['lr'][level]
    elif dream_config['lr_mode'] == 'ratio':
        start_lr, lr_ratio = dream_config['lr']
        return start_lr * lr_ratio ** level
    else:
        return dream_config['lr']

class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """

    def __init__(self, kernel_size, sigma, device='cuda'):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        # std multipliers, hardcoded to use 3 different Gaussian kernels
        cascade_coefficients = [0.5, 1.0, 2.0]
        sigmas = [[coeff * sigma, coeff * sigma]
                  for coeff in cascade_coefficients]  # isotropic Gaussian

        # assure we have the same spatial resolution
        self.pad = int(kernel_size[0] / 2)

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * \
                    torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(device)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad,
                              self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3
