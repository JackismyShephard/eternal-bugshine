from IPython.display import clear_output

import numbers
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from .utils.visual import reshape_image, get_noise_image, tensor_to_image, show_img, postprocess_image, make_video, image_to_tensor, random_shift, save_img
from .utils.config import extend_path, save_config

def dream_process(config, img = None):
    if img is not None:
        pass
    elif config['input_img_path'] is not None:
        img = cv.imread(config['input_img_path'])[:, :, ::-1]
        img = reshape_image(img, config['target_shape'])
        img = img.astype(np.float32)  # convert from uint8 to float32
        img /= 255.0  # get to [0, 1] range
    else:
        img = get_noise_image(config['noise'], config['target_shape'])
    
    img = (img - config['mean']) / config['std']
    output_images = dreamspace(img, config['model'], config)

    if config['output_img_path'] is not None:
        path = extend_path(config['output_img_path'], config['img_overwrite'])
        save_config(config, path)
        #show_img(output_images[-1], figsize=config['figsize'], show_axis='off',
        #         dpi=config['dpi'], save_path=path, close = True)
        save_img(output_images[-1], path)

    if config['video_path'] is not None:
        path = extend_path(config['video_path'], config['video_overwrite'])
        save_config(config, path)
        make_video(output_images, config['target_shape'], path)
    
    return output_images

def dreamspace(img, model, config):
    output_images = []
    start_size = img.shape[:-1]  # save initial height and width

    for level in range(config['levels']):
        scaled_tensor = scale_level(img, start_size, level,
                                    config['ratio'], config['levels'],  config['device'])

        for i in range(config['num_iters']):
            h_shift, w_shift = np.random.randint(-config['shift_size'], config['shift_size'] + 1, 2)
            shifted_tensor = random_shift(scaled_tensor, h_shift, w_shift, requires_grad = True)
            dreamt_tensor = dream_ascent(shifted_tensor, model, i, config)
            deshifted_tensor = random_shift(dreamt_tensor, h_shift,
                                             w_shift, undo=True, requires_grad = True)

            img = tensor_to_image(deshifted_tensor)
            output_image = postprocess_image(img, config['mean'],config['std'])
            if config['show'] == True:
                clear_output(wait=True)
                show_img(output_image, figsize=config['figsize'], show_axis='off', 
                            dpi=config['dpi'])

            if (i % config['save_interval']) == 0:
                output_images.append(output_image)

            scaled_tensor = deshifted_tensor
    
    return output_images


def scale_level(img, start_size, level, ratio=1.8,
                levels=4, device='cuda'):
    exponent = level - levels + 1
    h, w = np.round(np.float32(start_size) *
                    (ratio ** exponent)).astype(np.int32)
    scaled_img = cv.resize(img, (w, h))
    scaled_tensor = image_to_tensor(scaled_img, device, requires_grad=True)
    return scaled_tensor

def dream_ascent(tensor, model, iter, config):
    ## get activations
    activations = model(tensor, config['out_info'])
    ### calculate loss on desired layers
    losses = []
    for layer_activation in activations.values():
        if config['loss_type'] == 'norm':
            loss_part = torch.linalg.norm(layer_activation)
        elif config['loss_type'] == 'mean':
            loss_part = torch.mean(layer_activation)
        else:
            MSE = torch.nn.MSELoss(reduction='mean')
            zeros = torch.zeros_like(layer_activation)
            loss = MSE(layer_activation, zeros)
        losses.append(loss_part)
    if config['loss_red'] == 'mean':
        loss = torch.mean(torch.stack(losses))
    else:
        loss = torch.sum(torch.stack(losses))
    # do backpropagation and get gradient
    loss.backward()
    grad = tensor.grad.data
    ### gaussian smoothing
    if config['smooth'] == True:
        sigma = ((iter + 1) / config['num_iters']
                 ) * 2.0 + config['smooth_coef']
        smooth_grad = CascadeGaussianSmoothing(
            kernel_size=config['kernel_size'], sigma=sigma,
            device=config['device'])(grad)
    else:
        smooth_grad = grad
    ### normalization of gradient
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    if config['norm_type'] == 'standardize':
        smooth_grad = smooth_grad - g_mean
        smooth_grad = smooth_grad / g_std
    else:
        smooth_grad /= torch.abs(g_mean + config['eps'])

    ### gradient update ####
    tensor.data += config['lr'] * smooth_grad
    tensor.grad.data.zero_()
    ### clamp gradient to avoid it diverging. vanishing/exploding gradient phenomenon?
    if config['clamp_type'] == 'standardize':
        image_min = torch.tensor(
            (-config['mean'] / config['std']).reshape(1, -1, 1, 1)).to(config['device'])
        image_max = torch.tensor(
            ((1 - config['mean']) / config['std']).reshape(1, -1, 1, 1)).to(config['device'])
    elif config['clamp_type'] == 'unit':
        image_min, image_max = 0, 1
    else:
        image_min, image_max = -1, 1
    tensor.data = torch.clip(tensor, image_min, image_max)
    return tensor


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
