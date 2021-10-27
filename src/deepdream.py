

import numbers
import math

import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .utils.config import add_info_to_path, save
from .utils.custom_types import *
from .utils.visual import reshape_image, get_noise_image, tensor_to_image, postprocess_image, image_to_tensor, random_shift, save_img, Rendering, save_video



def dream_process(model : torch.nn.Module, dream_config : DreamConfig, model_config : ModelConfig, 
                    dataset_config : DatasetConfig, training_config : TrainingConfig, 
                    img : t.Optional[npt.NDArray[np.float32]] = None) -> t.List[npt.NDArray[np.uint8]]:
    if img is not None:
        input_img = img
    elif dream_config['input_img_path'] is not None:
        input_img = cv.imread(dream_config['input_img_path'])[:, :, ::-1]
        input_img = reshape_image(input_img, dream_config['target_shape'])
        input_img = input_img.astype(np.float32)  # convert from uint8 to float32
        input_img /= np.array(255.0)  # get to [0, 1] range
    elif dream_config['noise'] is not None:
        input_img = get_noise_image(dream_config['noise'], dream_config['target_shape'], 
                                dream_config['correlation'], dream_config['correlation_std'], 
                                dream_config['noise_scale'])
        input_img = input_img.astype(np.float32)
        input_img = (input_img - input_img.min())/(input_img.max() - input_img.min() )
    else:
        raise RuntimeError('img, input_img_path and noise are all None')
    input_img = (input_img - dream_config['mean']) / dream_config['std']
    output_images = dreamspace(input_img, model, dream_config, model_config['device'])
    if dream_config['add_path_info']:
        path_info = model_config['model_name']
        for (layer, idxs) in dream_config['target_dict'].items():
            path_info =  path_info + '_' + layer + '_' + str(idxs)

    else:
        path_info = None
    if dream_config['output_img_path'] is not None:
        path = add_info_to_path(dream_config['output_img_path'], 
            path_info, dream_config['output_img_ext'], dream_config['img_overwrite'])
        save(path, model_config, dataset_config, training_config,  dream_config = dream_config)
        save_img(output_images[-1], path)

    if dream_config['video_path'] is not None:
        path = add_info_to_path(dream_config['video_path'], path_info, 
                                    dream_config['video_ext'], dream_config['video_overwrite'])
        save(path, model_config, dataset_config, training_config,dream_config = dream_config)
        save_video(path, output_images, dream_config['target_shape'])
    
    return output_images

#TODO figure out if rescaling leaves artifacts in output image
def scale_level(img: npt.NDArray[t.Any], start_size: t.Tuple, level: int,
                ratio: float = 1.8, levels: int = 4, 
                gauss_filter : t.Optional[t.Tuple[int, int, float, float]] = None) -> npt.NDArray[t.Any]:

    exponent = level - levels + 1
    h, w = np.round(np.float32(np.array(start_size)) *
                    (ratio ** exponent)).astype(np.int32)
    if (h < img.shape[0]):
        #interpolation_mode = cv.INTER_AREA
        interpolation_mode = cv.INTER_LINEAR_EXACT
        if gauss_filter is not None:
            img = cv.GaussianBlur(img, ksize = gauss_filter[0:2] ,
                                  sigmaX=gauss_filter[2], sigmaY=gauss_filter[3], borderType=cv.BORDER_REFLECT)
        scaled_img = cv.resize(img, (w, h), interpolation=interpolation_mode)
    else:
        interpolation_mode = cv.INTER_LINEAR_EXACT
        #interpolation_mode = cv.INTER_CUBIC
        scaled_img = cv.resize(img, (w, h), interpolation=interpolation_mode)
        if gauss_filter is not None:
            scaled_img = cv.GaussianBlur(img, ksize=gauss_filter[0:2],
                                         sigmaX=gauss_filter[2], sigmaY=gauss_filter[3], borderType=cv.BORDER_REFLECT)
    return scaled_img


def dreamspace(img : npt.NDArray[np.float32], model : torch.nn.Module, 
                    dream_config : DreamConfig, device : torch.device) -> t.List[npt.NDArray[np.uint8]]:
    output_images = []
    start_size = img.shape[:-1]  # save initial height and width
    
    if dream_config['show'] == True:
        render = Rendering(dream_config['target_shape'])

    for level in range(dream_config['levels']):
        scaled_img = scale_level(img, start_size, level,
                                    dream_config['ratio'], dream_config['levels'])
        scaled_tensor = image_to_tensor(scaled_img, device, requires_grad=True)

        for i in range(dream_config['num_iters']):
            h_shift, w_shift = np.random.randint(-dream_config['shift_size'], dream_config['shift_size'] + 1, 2)
            shifted_tensor = random_shift(scaled_tensor, h_shift, w_shift, requires_grad = True)
            dreamt_tensor = dream_ascent(shifted_tensor, model, i, dream_config, device)
            deshifted_tensor = random_shift(dreamt_tensor, h_shift,
                                             w_shift, undo=True, requires_grad = True)

            img = tensor_to_image(deshifted_tensor)
            output_image = postprocess_image(img, dream_config['mean'],dream_config['std'])
            if dream_config['show'] == True:
                render.update(output_image)

            if (i % dream_config['save_interval']) == 0:
                output_images.append(output_image)

            scaled_tensor = deshifted_tensor
    return output_images

def dream_ascent(tensor : torch.Tensor, model : torch.nn.Module, iter : int, 
                dream_config : DreamConfig, device : torch.device) -> torch.Tensor:
    ## get activations
    _, activations = model(tensor, dream_config['target_dict'])
    ### calculate loss on desired layers
    losses = []
    for layer_activation in activations:
        if dream_config['loss_type'] == 'norm':
            loss_part = torch.linalg.norm(layer_activation)
        elif dream_config['loss_type'] == 'mean':
            loss_part = torch.mean(layer_activation)
        else:
            MSE = torch.nn.MSELoss(reduction='mean')
            zeros = torch.zeros_like(layer_activation)
            loss_part = MSE(layer_activation, zeros)
        losses.append(loss_part)
    if dream_config['loss_red'] == 'mean':
        loss = torch.mean(torch.stack(losses))
    else:
        loss = torch.sum(torch.stack(losses))
    # do backpropagation and get gradient
    loss.backward()
    grad = tensor.grad.data
    ### gaussian smoothing
    sigma = ((iter + 1) / dream_config['num_iters']) * 2.0 + dream_config['smooth_const']
    smooth_grad = gradient_smoothing(grad, dream_config['kernel_size'], sigma, dream_config['smooth_factors'])
    #smooth_grad = CascadeGaussianSmoothing(kernel_size=dream_config['kernel_size'], sigma=sigma,device=device)(grad)
    ### normalization of gradient
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    if dream_config['norm_type'] == 'standardize':
        smooth_grad = smooth_grad - g_mean
        smooth_grad = smooth_grad / g_std
    else:
        smooth_grad /= torch.abs(g_mean + dream_config['eps'])

    ### gradient update ####
    tensor.data += dream_config['lr'] * smooth_grad
    tensor.grad.data.zero_()
    ### clamp gradient to avoid it diverging. vanishing/exploding gradient phenomenon?
    if dream_config['clamp_type'] == 'standardize':
        image_min = torch.tensor(
            (-dream_config['mean'] / dream_config['std']).reshape(1, -1, 1, 1)).to(device)
        image_max = torch.tensor(
            ((1 - dream_config['mean']) / dream_config['std']).reshape(1, -1, 1, 1)).to(device)
    elif dream_config['clamp_type'] == 'unit':
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
