from IPython.display import clear_output

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio

import torch
from torchvision import transforms


from src.deepdream_aux import scale_level, random_shift, output_adapter, CascadeGaussianSmoothing

from src.utils.visual import show_tensor

def dreamspace(image, model, config):
    output_tensors = []
    (w, h) = image.size
    start_size = (h, w)
    if config['use_noise'] == 'uniform':
        (h, w) = config['noise_shape']
        image = np.random.uniform(size=(h, w, 3)).astype(np.float32)
    elif config['use_noise'] == 'gaussian':
        (h, w) = config['noise_shape']
        image = np.random.normal(size=(h, w, 3)).astype(np.float32)
    tensorfy = transforms.ToTensor()
    normalize = transforms.Normalize(config['mean'], config['std'])
    dream_transform = transforms.Compose([tensorfy, normalize])
    tensor = dream_transform(image)

    for level in range(config['levels']):
        scaled_tensor = scale_level(tensor, start_size, level,
                                    config['ratio'], config['levels'],  config['device'])
        for i in range(config['num_iters']):
            if config['shift'] == True:
                h_shift, w_shift = np.random.randint(
                    -config['shift_size'], config['shift_size'] + 1, 2)
                scaled_tensor = random_shift(scaled_tensor, h_shift, w_shift)
            dreamt_tensor = dream_ascent(scaled_tensor, model, i, config)
            if config['shift'] == True:
                dreamt_tensor = random_shift(dreamt_tensor, h_shift,
                                             w_shift, undo=True)
            if config['show'] == True:
                clear_output(wait=True)
                show_tensor(output_adapter(dreamt_tensor), config['mean'],
                            config['std'], figsize=config['figsize'], show_axis='off')
            if (i % config['save_interval']) == 0:
                output_tensors.append(dreamt_tensor)

            scaled_tensor = dreamt_tensor
        tensor = output_adapter(dreamt_tensor)
    if config['img_path'] is not None:
        clear_output(wait=True)
        show_tensor(output_adapter(output_tensors[-1]),
                    config['mean'], config['std'],
                    save_path=config['img_path'], dpi=config['dpi'],
                    figsize=config['figsize'], show_axis='off')
    if config['video_path'] is not None:
        Path('videos/temp/').mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(config['video_path'], mode='I') as writer:
            for i, tensor in enumerate(output_tensors):
                clear_output(wait=True)
                show_tensor(output_adapter(tensor), config['mean'], config['std'],
                            save_path='videos/temp/'+str(i) + '.jpg',
                            dpi=200, close=True, figsize=config['figsize'], show_axis='off')
                image = imageio.imread('videos/temp/'+str(i) + '.jpg')
                writer.append_data(image)
                os.remove('videos/temp/'+str(i) + '.jpg')
        writer.close()
    return output_tensors


def dream_ascent(tensor, model, iter, config):
    ## get activations
    activations = model(tensor, config['out_info'])
    ### calculate loss on desired layers
    losses = []
    for layer_activation in activations.values():
        if config['loss_type'] == 'norm':
            loss = torch.linalg.norm(layer_activation)
        elif config['loss_type'] == 'mean_red':
            loss = torch.mean(layer_activation)
        else:
            MSE = torch.nn.MSELoss(reduction='mean')
            zeros = torch.zeros_like(layer_activation)
            loss = MSE(layer_activation, zeros)
        losses.append(loss)
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
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std
    ### gradient update ####
    tensor.data += config['lr'] * smooth_grad
    tensor.grad.data.zero_()
    ### clamp gradient to avoid it diverging. vanishing/exploding gradient phenomenon?
    if config['clamp_type'] == 'norm':
        image_min = torch.tensor(
            (-config['mean'] / config['std']).reshape(1, -1, 1, 1)).to(config['device'])
        image_max = torch.tensor(
            ((1 - config['mean']) / config['std']).reshape(1, -1, 1, 1)).to(config['device'])
    else:
        image_min, image_max = -1, 1
    tensor.data = torch.clip(tensor, image_min, image_max)
    return tensor