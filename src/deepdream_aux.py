import numbers
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.utils.headers import IMAGENET_MEAN, IMAGENET_STD

def input_adapter(tensor, device='cuda:0'):
    tensor = tensor.to(device).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

def output_adapter(tensor):
    return tensor.to('cpu').detach().squeeze(0)


def scale_level(tensor, start_size, level, ratio=1.8,
                levels=4, device = 'cuda:0'):
    exponent = level - levels + 1
    h, w = np.int32(np.float32(start_size) * (ratio ** exponent))
    # we resize tensors directly instead of PIL images.
    # The difference here is that resizing on PIL images does automatic antialiasing
    # Note that the cv.resize used elsewhere does not use antialiasing
    scale_tensor = transforms.Resize((h, w))(tensor)
    scale_tensor = input_adapter(scale_tensor, device)
    return scale_tensor


def scale_space(image, ratio=1.8, levels=4, mean=IMAGENET_MEAN, 
                std=IMAGENET_STD, device='cuda:0'):
    scaled_tensors = []
    x, y = image.size
    tensorfy = transforms.toTensor()
    normalize = transforms.Normalize(mean = mean, std = std)
    dream_transform = transforms.Compose([tensorfy, normalize])
    tensor = dream_transform(image)
    for level in range(levels):
        scaled_tensor = scale_level(tensor, (y, x), level, ratio, device=device)
        scaled_tensors.append(scaled_tensor)
    return scaled_tensors

def random_shift(tensor, h_shift, w_shift, undo=False):
    if undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = True
        return rolled


class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).

    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.

    """

    def __init__(self, kernel_size, sigma, device='cuda:0'):
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
