import collections
import matplotlib.pyplot as plt
import matplotlib.transforms as plt_transform
import matplotlib.colors as mcolors
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
from PIL import Image
from skimage import filters as skfilt
from skimage.util import random_noise

# used for image rendering
import io
from ipywidgets import widgets
from IPython import display

from .config import BEETLENET_MEAN, BEETLENET_STD, RNG_SEED, DEVICE
from .custom_types import PlotConfig


def plot_metrics(plot_config: PlotConfig, x, metrics, save_path=None):
    fig, ax = plt.subplots(plot_config['fig_row'],plot_config['fig_column'],
                             figsize=(plot_config['size_w'],plot_config['size_h']))

    titles = plot_config['titles']
    y_label = plot_config['y_label']

    if plot_config['use_title_label']:
        y_label = titles

    # list of colors to make the metric and the average the same color
    colors = list(mcolors.TABLEAU_COLORS)

    # container to use a single loop for all plots
    ax_container = np.array(ax).reshape(-1)

    # metric is packed as ['metric', 'average']
    step = 2
    
    for i in range(ax_container.shape[0]):
        for j in range(metrics[i].shape[0]//2):
            ax_container[i].plot(x, metrics[i,j*step], color = colors[j%len(colors)],
                             label=plot_config['label'][j],
                            linestyle=plot_config['param_linestyle'], alpha=plot_config['param_alpha'])
        
            if plot_config['show_rolling_avg']:
                ax_container[i].plot(x, metrics[i,j*step + 1], color = colors[j%len(colors)], 
                                label = plot_config['label'][j] + ' ' + plot_config['rolling_avg_label'],
                                linestyle=plot_config['average_linestyle'])

        ax_container[i].set_xlabel(plot_config['x_label'])
        ax_container[i].set_ylabel(titles[i])
        ax_container[i].legend()

        if plot_config['show_grid']:
            ax_container[i].grid()
        
        if plot_config['show_title']:
            ax_container[i].set_title(titles[i])
        
    plt.tight_layout()
    if plot_config['save_figure']:
        if plot_config['save_subfigures']:
            for i in range(ax_container.shape[0]):
                index_y = i // plot_config['fig_column']
                index_x = i % plot_config['fig_column']

                sub_h = plot_config['size_h'] / plot_config['fig_row']
                sub_w = plot_config['size_w'] / plot_config['fig_column']
 
                area = plt_transform.Bbox([ [index_x*sub_w,index_y*sub_h],
                                        [(index_x+1)*sub_w,(index_y+1)*sub_h]])

                file_path = save_path + '_' + titles[i].lower() + '_' + plot_config['save_extension']
                fig.savefig(file_path + '.pdf',  bbox_inches=area,
                            facecolor='w', dpi=plot_config['save_dpi'])
                if plot_config['save_copy_png']:
                    fig.savefig(file_path + '.png',  bbox_inches=area,
                            facecolor='w', dpi=plot_config['save_dpi'])
        else:
            file_path = save_path + '_' + plot_config['save_extension']
            fig.savefig(file_path + '.pdf',  bbox_inches='tight',
                        facecolor='w', dpi=plot_config['save_dpi'])
            if plot_config['save_copy_png']:
                fig.savefig(file_path + '.png',  bbox_inches='tight',
                        facecolor='w', dpi=plot_config['save_dpi'])


    plt.show()
    plt.close()


# TODO add else branch returning error
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

# TODO sigma in skfilt.gaussian and scale in np.random.normal should be parameterized
# QUESTION why are we using mode = 'reflect'?
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
    elif color == 'black':
        _color = [0.,0.,0.]
    elif color == 'red':
        _color = [1.,0.,0.]
    elif color == 'green':
        _color = [0.,1.,0.]
    elif color == 'blue':
        _color = [0.,0.,1.]
    (h, w) = shape
    img = np.zeros(shape=(h,w,3), dtype=np.float32)
    for i in range(3):
        img[:,:,i] = _color[i]
    return img

def add_noise(type, img):
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

#TODO consider using our transform.ToTensor wrapper class
def image_to_tensor(img, device=DEVICE, requires_grad=False):
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


# TODO implement more features and make it more dynamic
# TODO make it more pretty
# TODO start image shape dims and scaling of new image shape should be parameterized
# TODO add a more comprehensive explanation of the function
class Rendering():
    """
        Class for rendering dreamt images.
    """


    def __init__(self, shape):
        self.format = 'png'
        start_image = np.full((200,400,3), 255).astype(np.uint8)
        image_stream = self.compress_to_bytes(start_image)

        h, w = shape
        self.widget = widgets.Image(value = image_stream, width=w*2, height=h*2)
        # QUESTION where is display loaded? It should be in this module.
        display(self.widget)

    # To display the images, they need to be converted to a stream of bytes
    def compress_to_bytes(self, data):
        """
            Helper function to compress image data via PIL/Pillow.
        """
        buff = io.BytesIO()
        img = Image.fromarray(data)    
        img.save(buff, format=self.format)
    
        return buff.getvalue()

    def update(self, image):
        stream = self.compress_to_bytes(image)
        self.widget.value = stream

