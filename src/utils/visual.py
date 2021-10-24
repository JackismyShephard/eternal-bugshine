import matplotlib.pyplot as plt
import matplotlib.transforms as plt_transform
import matplotlib.colors as mcolors

import numpy as np
import cv2 as cv
from PIL import Image

import torch
from torchvision import transforms

from skimage import filters as skfilt
from skimage.util import random_noise

import typing as t
from numpy import float32, typing as npt

# used for image rendering
import io
from ipywidgets import widgets

from .config import BEETLENET_MEAN, BEETLENET_STD, DEVICE
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


def reshape_image(img: npt.NDArray[t.Any], shape: t.Union[int, t.Tuple[int, int]]) -> npt.NDArray[t.Any]:
    current_height, current_width = img.shape[:2]
    if isinstance(shape, int):
        new_height = int(current_height * (shape / current_width))
        new_width = current_width
    else:
        new_height, new_width = shape[:2]

    if new_height * new_width < current_height * current_width :
        interpolation_mode = cv.INTER_AREA
    else:
        interpolation_mode = cv.INTER_CUBIC

    new_img = cv.resize(img, (new_width, new_height),interpolation=interpolation_mode)

    return new_img



# QUESTION why are we using mode = 'reflect'?
def get_noise_image(type: t.Literal['uniform', 'gaussian'], shape: t.Union[int, t.Tuple[int, int]],
                    correlation: t.Optional[str] = None, sigma=1.0, scale = 1.0, 
                    low = 0.0, high = 1.0, loc = 0.0) -> npt.NDArray[t.Any]:
    if isinstance(shape, int):
        h,w = shape, shape
    else:
        h,w = shape

    if type == 'uniform':
        img = np.random.uniform(low, high, size=(h, w, 3))
    else:
        img = np.random.normal(loc, scale, size=(h, w, 3))
    
    if correlation == 'gaussian':
            img = skfilt.gaussian(img, sigma, mode='reflect', multichannel=True)
    else:
        pass
    return img

def add_noise(type: str, img: npt.NDArray) -> t.Any:
    return random_noise(img, type)  


def preprocess_image(img: npt.NDArray[t.Any], mean: npt.NDArray[np.float32] = BEETLENET_MEAN,
                     std: npt.NDArray[np.float32]= BEETLENET_STD, range = 255.0) -> npt.NDArray[np.float32]:
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= range  # get to [0, 1] range
    img = (img - mean) / std
    return img


def postprocess_image(img: npt.NDArray[t.Any], mean: npt.NDArray[np.float32] = BEETLENET_MEAN, 
                        std: npt.NDArray[np.float32] = BEETLENET_STD) -> npt.NDArray[np.uint8]:
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = ((img*255).astype(np.uint8))
    return img


def image_to_tensor(img: npt.NDArray[t.Any], device: torch.device = DEVICE, 
                        requires_grad: bool = False) -> torch.Tensor:
    tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
    tensor.requires_grad = requires_grad
    return tensor

def tensor_to_image(tensor : torch.Tensor) -> npt.NDArray[t.Any]:
    tensor = tensor.to('cpu').detach().squeeze(0)
    img = tensor.numpy().transpose((1, 2, 0))
    return img

def random_shift(tensor : torch.Tensor, h_shift : int, w_shift : int, 
                    undo : bool =False, requires_grad : bool = True) -> torch.Tensor:
    if undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = requires_grad
        return rolled


def show_img(img : npt.NDArray[t.Any], title: t.Optional[str] = None, save_path : t.Optional[str]=None, 
                dpi : t.Union[float]  =200, figsize : t.Tuple[float, float]=(7, 7), 
                show_axis : str ='on', close : bool =False) -> None:
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

def save_img(img : npt.NDArray[t.Any], path : str) -> None:
    cv.imwrite(path, img[:, :, ::-1])


def make_video(images: t.List[npt.NDArray[t.Any]], shape : t.Union[int, t.Tuple[int, int]], path : str) -> None:
    imgs = [Image.fromarray(reshape_image(img, shape)) for img in images]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0)


# TODO implement more features and make it more dynamic
# TODO make it more pretty
class Rendering():
    """
        Class for rendering dreamt images.
    """


    def __init__(self, shape=(200,400), scale = 2):
        self.format = 'png'
        h, w = shape
        start_image = np.full((h,w,3), 255).astype(np.uint8)
        image_stream = self.compress_to_bytes(start_image)

        self.widget = widgets.Image(value = image_stream, width=w*scale, height=h*scale)
        # QUESTION where is display loaded? It should be in this module.
        display(self.widget) 

    # To display the images, they need to be converted to a stream of bytes
    def compress_to_bytes(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
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

