

# used for image rendering

import io
from ipywidgets import widgets

import matplotlib.pyplot as plt
import matplotlib.transforms as plt_transform
import matplotlib.colors as mcolors

import numpy as np
import cv2 as cv
import imageio
from PIL import Image

import torch
from torchvision import transforms

from skimage import filters as skfilt
from skimage.util import random_noise

import typing as t
from numpy import float32, typing as npt

from .config import BEETLENET_MEAN, BEETLENET_STD, DEVICE
from .custom_types import PlotConfig

def plot_metrics(plot_config: PlotConfig, x : npt.NDArray[t.Any], metrics : npt.NDArray[t.Any], 
                        save_path : t.Optional[str] =None) -> None:
    fig, ax = plt.subplots(plot_config['fig_row'],plot_config['fig_column'],
                             figsize=(plot_config['size_w'],plot_config['size_h']))

    titles = plot_config['titles']

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
                if plot_config['rolling_avg_label'] is None:
                    raise RuntimeError('Trying to plot average metrics but no corresponding label was given')
                else:
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
        if save_path is None:
            raise RuntimeError('Trying to save figure but no save path was given')
        elif plot_config['save_subfigures']:
            for i in range(ax_container.shape[0]):
                index_y = i // plot_config['fig_column']
                index_x = i % plot_config['fig_column']

                sub_h = plot_config['size_h'] / plot_config['fig_row']
                sub_w = plot_config['size_w'] / plot_config['fig_column']
 
                area = plt_transform.Bbox([ [index_x*sub_w,index_y*sub_h],
                                        [(index_x+1)*sub_w,(index_y+1)*sub_h]])

                file_path = save_path + '_' + titles[i].lower() + '_' + plot_config['save_suffix']
                fig.savefig(file_path + plot_config['save_ext'],  bbox_inches=area,
                            facecolor='w', dpi=plot_config['save_dpi'])
        else:
            file_path = save_path + '_' + plot_config['save_suffix']
            fig.savefig(file_path + plot_config['save_ext'],  bbox_inches='tight',
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
        #interpolation_mode =  cv.INTER_AREA
        interpolation_mode = cv.INTER_LINEAR_EXACT
    else:
        interpolation_mode = cv.INTER_LINEAR_EXACT
        #interpolation_mode = cv.INTER_CUBIC

    new_img = cv.resize(img, (new_width, new_height),interpolation=interpolation_mode)

    return new_img


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
    tensor = transforms.ToTensor()(img).type(torch.float32).to(device).unsqueeze(0)
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
                dpi : float  =200, figsize : t.Tuple[float, float]=(7, 7), 
                show_axis : str ='on', close : bool =False) -> None:
    '''This is a utility function for displaying numpy images with matplotlib. Currently not used as widgets are better'''
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




def save_video(path : str, images: t.List[npt.NDArray[t.Any]], shape: t.Union[int, t.Tuple[int, int]], quality : int = 7, 
                fps: int = 24, macro_block_size :int  = 1) -> None:
    '''Creates a video from a list of images. The type of video format is inferred from the "path" parameter. The "shape" parameter
        defines the output size of the video. The parameter "quality" controls the quality and thereby also the size of the created video. With quality = 7 files of size around 1MB are created. '''
    outputdata = np.array([reshape_image(img, shape) for img in images])
    imageio.mimwrite(path, outputdata, quality=quality,
                     macro_block_size=macro_block_size, fps=fps)

def save_video2(path, images: t.List[npt.NDArray[t.Any]], shape: t.Union[int, t.Tuple[int, int]], fps: int = 24) -> None:
    '''This is an attempt at saving mp4 files using opencv. Currently not working.'''
    reshaped_imgs = [cv.cvtColor(reshape_image(
        img, shape), cv.COLOR_RGB2BGR) for img in images]
    (h, w, _) = reshaped_imgs[0].shape
    print(shape)
    out = cv.VideoWriter(path, cv.VideoWriter_fourcc(
        'M', 'P', 'E', 'G'), fps, (w, h), False)
    for image in reshaped_imgs:
        out.write(image)
    out.release()

def save_video3(path: str, images: t.List[npt.NDArray[t.Any]], shape: t.Union[int, t.Tuple[int, int]]) -> None:
    '''This is the old model for saving videos as gif using PIL. Creates very large files.'''
    imgs = [Image.fromarray(reshape_image(img, shape)) for img in images]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0)

import time
# TODO implement more features and make it more dynamic
# TODO make it more pretty
class Rendering():
    """
        Class for rendering dreamt images.
    """

    def __init__(self, shape : t.Union[t.Tuple[int, int], int]=(200,400), scale :int = 2) -> None:
        self.format = 'png'
        if isinstance(shape, int):
            h, w =  (shape, shape)
        else:
            h, w = shape

        start_image = np.full((h,w,3), 255).astype(np.uint8)
        image_stream = self.compress_to_bytes(start_image)
        self.widget = widgets.Image(value = image_stream, width=w*scale, height=h*scale)
        display(self.widget) 


    # To display the images, they need to be converted to a stream of bytes
    def compress_to_bytes(self, data : t.Union[npt.NDArray[t.Any], torch.Tensor]) -> bytes:
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        """
            Helper function to compress image data via PIL/Pillow.
        """
        buff = io.BytesIO()
        img = Image.fromarray(data)    
        img.save(buff, format=self.format)
    
        return buff.getvalue()

    def update(self, image : npt.NDArray[t.Any]) -> None:
        stream = self.compress_to_bytes(image)
        time.sleep(1/144)
        self.widget.value = stream

class Rendering_stats(widgets.VBox):
    def __init__(self, shape, scale = 2, target_dict = {'fc' : 0}):
        super().__init__()        

        self.target = 0
        self.data = np.zeros(197)
        self.loss_target = 0
        self.loss_penalty = 0
        self.max_y = 50
        self.min_y = -50
        # setup image
        self.format = 'png'
        if isinstance(shape, int):
            h, w =  (shape, shape)
        else:
            h, w = shape
        start_image = np.full((h,w,3), 255).astype(np.uint8)
        image_stream = self.compress_to_bytes(start_image)

        self.image = widgets.Image(value = image_stream, width=w*scale, height=h*scale)
        self.image.layout = widgets.Layout(
                border='solid 1px black',
                margin='0px 10px 10px 0px',
                padding='5px 5px 5px 5px')
 
        # create plot in widget
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(1,1,constrained_layout=True, figsize=(12, 4))

        # plot setup
        self.line, = self.ax.plot([0,197], [0,0], 'b')
        self.target_point = self.ax.scatter([-50],[0], color = 'g', label='target')
        self.current_point = self.ax.scatter([-50],[0], color = 'r', label='current prediction')
        self.ax.set_ylim([self.min_y, self.max_y])
        self.ax.set_xlim([-5,197+5])
        self.ax.set_title('Model response')
        self.ax.set_xlabel('Class')
        self.ax.grid(True)
        self.ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.40),
          ncol=2)

        self.fig.canvas.toolbar_visible = False

        
        # create labels for the statistics
        self.loss_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))
        self.loss_target_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))
        self.loss_penalty_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))

        self.value_target_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))
        self.highest_value_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))

        self.target_class_label = widgets.Label(str(self.target), layout=widgets.Layout(display="flex", justify_content="center"))
        self.current_class_label = widgets.Label('0', layout=widgets.Layout(display="flex", justify_content="center"))


        # Place each statistic field in a layout
        self.loss_box = self.create_label('Current loss : ', self.loss_label)

        loss_target_box = self.create_label('Target loss : ', self.loss_target_label)
        loss_penalty_box = self.create_label('Penalty loss : ', self.loss_penalty_label)
        self.losses_box = widgets.HBox([loss_target_box,   loss_penalty_box ])
        self.losses_box.layout = widgets.Layout(display='flex', align_items='flex-start',width='90%')

        values_target_box = self.create_label('Target value : ', self.value_target_label)
        highest_value_box = self.create_label('Predicted value : ', self.highest_value_label)
        self.values_box = widgets.HBox([values_target_box,   highest_value_box ])
        self.values_box.layout = widgets.Layout(display='flex', align_items='flex-start',width='90%')

        target_class_box = self.create_label('Target class : ', self.target_class_label)
        predicted_class_box = self.create_label('Predicted class : ', self.current_class_label)
        self.class_box = widgets.HBox([target_class_box,   predicted_class_box ])
        self.class_box.layout = widgets.Layout(display='flex', align_items='flex-start',width='90%')


        # Place all statistics in its own layout
        controls = widgets.VBox([
            self.loss_box,
            self.losses_box, 
            self.values_box,
            self.class_box,
        ])
        controls.layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='50%',
                border='solid 1px black',
                margin='0px 10px 10px 0px',
                padding='5px 5px 5px 5px')


        # Place both image and the statistics in the top part
        top = widgets.HBox([self.image, controls])

        top.layout = widgets.Layout(display='flex',
                flex_flow='row',
                align_items='center',
                border='solid 1px black',
                margin='0px 10px 10px 0px',
                padding='5px 5px 5px 5px')

        # Place graph in its own layout
        out_box = widgets.Box([output])
        output.layout = widgets.Layout()
 
        # add to children
        self.children = [top, out_box]

        #display(self)
    
    def create_label(self, label, pointer, value=""):
        """Helper function to create frame and label for each field"""
        text = widgets.Label(label, layout=widgets.Layout(display="flex", justify_content="center"))
        box = widgets.HBox([text, pointer])
        box.layout = widgets.Layout(
                display='flex',
                flex_flow='row',
                align_items='stretch',
                width='90%',
                border='solid 1px black',
                margin='0px 10px 10px 0px',
                padding='5px 5px 5px 5px')
        return box


    def update_graph(self, target, predicted):
        """Update plot"""
        x = np.linspace(0, self.data.shape[0]-1, self.data.shape[0])
        if np.max(self.data) > self.max_y:
            self.max_y = np.max(self.data) + 5

        if np.min(self.data) < self.min_y:
            self.min_y = np.min(self.data) - 5
        self.ax.set_ylim([self.min_y, self.max_y])
        self.line.set_xdata(x)
        self.line.set_ydata(self.data)
        self.target_point.set_offsets(target)
        self.current_point.set_offsets(predicted)
        self.fig.canvas.draw()


    def update_stats(self):
        """Update statistics"""
        # if penalty is used
        
        total_loss = self.loss_target - self.loss_penalty

        # set loss labels
        self.loss_label.value = str(format(total_loss, ".5g"))
        self.loss_target_label.value = str(format(self.loss_target, ".5g"))
        self.loss_penalty_label.value = str(format(self.loss_penalty, ".5g"))

        # calculate classes and position of points
        target_value = self.data[self.target]
        pred_value = np.max(self.data)
        predicted = np.argmax(self.data)

        # set remaining labels
        self.value_target_label.value = str(format(target_value, ".5g"))
        self.highest_value_label.value = str(format(pred_value, ".5g"))
        self.current_class_label.value = str(predicted)

        self.update_graph([self.target, target_value], [predicted, pred_value])

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
        """Update image"""
        stream = self.compress_to_bytes(image)
        self.image.value = stream
        self.update_stats()
        

    def __del__(self):
        self.fig.close()

