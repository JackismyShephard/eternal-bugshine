from PIL import Image
import typing as t
import numpy as np
from numpy import typing as npt
import torch
import torchvision.transforms
from torchvision.transforms import functional, ToTensor




class RandomizeBackground(torch.nn.Module):  
    """Replace beetle PIL image background color with a random color."""
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image): 
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255

        np_x_gray = (np.sum(np_x, axis=2)) / 3 
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        
        r = torch.rand(1).item()
        g = torch.rand(1).item()
        b = torch.rand(1).item()
        r_channel = np.ones(np_x_gray.shape) * r
        g_channel = np.ones(np_x_gray.shape) * g
        b_channel = np.ones(np_x_gray.shape) * b
        new_bg = np.dstack([r_channel, g_channel, b_channel])
        np_x = np.where(mask == True, new_bg, np_x)
        np_x = (np_x * 255).astype('uint8')
        return Image.fromarray(np_x)
    def __repr__(self) -> str:
        args = '[{}]'.format(self.cutoff)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'


class NotStupidRandomResizedCrop(torch.nn.Module):
    """A RandomResizedCrop reimplementation that does just what we need it to do.
        Crops a section of a PIL image with shape d*img.shape, where
        min_scale/100 <= d <= max_scale/100, at some random coordinate in the image."""

    def __init__(self, min_scale: float = 0.5, max_scale: float = 1) -> None:
        
        super().__init__()
        self.rng = np.random.default_rng() 
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.int_min_scale = int(min_scale * 100)
        self.int_max_scale = int(max_scale * 100)
    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img)
        scale = self.rng.integers(
            low=self.int_min_scale, high=self.int_max_scale) / 100
        (h, w, _) = np_x.shape
        height = int(scale * h)
        width = int(scale * w)
        x_pos = self.rng.random()
        y_pos = self.rng.random()
        y_max = h - height
        x_max = w - width
        left = int(x_pos * x_max)
        top = int(y_pos * y_max)
        img = functional.crop(img, top, left, height, width)
        
        img = functional.resize(img, [h,w]) 
        return img
    def __repr__(self) -> str:
        args = '[{},{}]'.format(self.min_scale, self.max_scale)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomizeBackgroundGraytone(torch.nn.Module):
    """Replace beetle PIL image background color with a random graytone."""
    def __init__(self, cutoff: float, min: float = 0, max: float = 1) -> None: 
        super().__init__()
        self.rng = np.random.default_rng()
        self.cutoff = cutoff
        self.min = min
        self.max = max
    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        color = self.rng.integers(int(self.min * 255), int(self.max * 255))
        np_x = np.where(mask == True, color, (np_x * 255).astype('uint8'))
        return Image.fromarray(np_x)
    def __repr__(self) -> str:
        args = '[{},{},{}]'.format(self.cutoff, self.min, self.max)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomizeBackgroundRGBNoise(torch.nn.Module):
    """Replace beetle PIL image background color with RGB noise."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.rng = np.random.default_rng()
        self.cutoff = cutoff    
    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        new_bg = self.rng.random(np_x.shape) 
        np_x = np.where(mask == True, new_bg, np_x)
        np_x = (np_x * 255).astype('uint8')
        return Image.fromarray(np_x)
    def __repr__(self) -> str:
        args = '[{}]'.format(self.cutoff)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'


def get_solid_color(color: t.Union[str, t.Tuple[float, float, float]],
                    shape: t.Union[int, t.Tuple[int, int]]) -> npt.NDArray[t.Any]:
    if color == 'white':
        _color = [1., 1., 1.]
    elif color == 'black':
        _color = [0., 0., 0.]
    elif color == 'red':
        _color = [1., 0., 0.]
    elif color == 'green':
        _color = [0., 1., 0.]
    elif color == 'blue':
        _color = [0., 0., 1.]
    else:
        _color = color

    if isinstance(shape, int):
        (h, w) = (shape, shape)

    elif isinstance(shape, tuple):
        h, w = shape

    else:
        raise TypeError('shape must be either an int or tuple')

    img = np.zeros(shape=(h, w, 3), dtype=np.float32)

    for i in range(3):
        img[:, :, i] = _color[i]
    return img

class CoarseDropout(torch.nn.Module):
    def __init__(self,  min_holes: int = 0, max_holes:int = 10, 
                        min_height: int = 5, max_height: int = 10, 
                        min_width:int = 5, max_width: int = 10, 
                        fill_type : t.Optional[t.Union[str, t.Tuple[float, float, float]]] = None) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.fill_type = fill_type

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255
        (h, w, _) = np_x.shape
        mask = np.ones(np_x.shape)
        holes = self.rng.integers(self.min_holes, self.max_holes)
        for _ in range(holes):
            width = self.rng.integers(self.min_width, self.max_width)
            height = self.rng.integers(self.min_height, self.max_height)
            x = self.rng.integers(0, w)
            y = self.rng.integers(0, h)
            mask[y:y+height,x:x+width,:] = 0

        if self.fill_type == 'random_rgb':
            fill = self.rng.random(np_x.shape)
        elif self.fill_type == 'random_uniform':
            r = np.ones((h,w)) * torch.rand(1).item()
            g = np.ones((h,w)) * torch.rand(1).item()
            b= np.ones((h,w)) * torch.rand(1).item()
            fill = np.dstack([r, g, b])
        elif self.fill_type is not None:
            fill = get_solid_color(self.fill_type, (h,w))
        else:
            fill = 0
        
        np_dropout = np.where(mask == 0, fill, np_x)
        np_dropout = (np_dropout * 255).astype('uint8')
        return Image.fromarray(np_dropout)
    def __repr__(self) -> str:
        args = '[{},{},{},{},{},{}, {}]'.format(self.min_holes, self.max_holes, 
        self.min_height, self.max_height, self.min_width, self.max_width, self.fill_type)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

#wrapper classes to load json representations of torchvision transforms. probably exists a smarter way, but im doing it
class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p: float=0.5) -> None:
        super().__init__()
        self.transform = torchvision.transforms.RandomVerticalFlip(p)
        self.p = p

    def forward(self, x: t.Union[Image.Image, torch.Tensor]) -> t.Union[Image.Image, torch.Tensor]:
        return self.transform(x)
    def __repr__(self) -> str:
        args = '[{}]'.format(self.p)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomRotation(torch.nn.Module):
    def __init__(self, degrees : t.Tuple[float, float] =(0,0), fill: float =255):
        super().__init__()
        self.transform = torchvision.transforms.RandomRotation(degrees, functional.InterpolationMode.NEAREST, False, None, fill, None)
        self.degrees = degrees
        self.fill = fill

    def forward(self, x: t.Union[Image.Image, torch.Tensor]) -> t.Union[Image.Image, torch.Tensor]:
        return self.transform(x)
    def __repr__(self) -> str:
        args = '[[{},{}],{}]'.format(self.degrees[0], self.degrees[1], self.fill)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class Resize(torch.nn.Module):
    def __init__(self, size : t.Tuple[int, int]=(200,200)) -> None:
        super().__init__()
        self.transform = torchvision.transforms.Resize(size, functional.InterpolationMode.BILINEAR, None, None)
        self.size = size

    def forward(self, x: t.Union[Image.Image, torch.Tensor]) -> t.Union[Image.Image, torch.Tensor]:
        return self.transform(x)
    def __repr__(self) -> str:
        args = '[[{},{}]]'.format(self.size[0], self.size[1])
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class ToTensor:
    def __init__(self) -> None:
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
    def __call__(self, x : Image.Image) -> torch.Tensor:
        return functional.to_tensor(x)
    def __repr__(self) -> str:
        args = '[]'
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class Normalize(torch.nn.Module):
    def __init__(self, mean: t.Union[t.Sequence[float], npt.NDArray[np.float32]], 
                 std: t.Union[t.Sequence[float], npt.NDArray[np.float32]], 
                 inplace: bool = False) -> None:
        super().__init__()
        self.transform = torchvision.transforms.Normalize(mean, std, inplace)
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def forward(self, x : torch.Tensor) -> torch.Tensor: 
        return self.transform(x)
    def __repr__(self) -> str:
        args = '[{}, {}]'.format(str(list(self.mean)), str(list(self.std)))
        return '{'+'"{}":'.format(self.__class__.__name__) +''+'{}'.format(args) + '}'


def string_to_class(transform_dict: t.Dict[str, t.Any]) -> t.Union[torch.nn.Module, ToTensor]:
    [(key, value)] = list(transform_dict.items())
    if key == 'RandomVerticalFlip':
        retval = RandomVerticalFlip(*value)
    elif key == 'RandomRotation':
        retval = RandomRotation(*value)
    elif key == 'Resize':
        retval = Resize(*value)
    elif key == 'ToTensor':
        retval = ToTensor()
    elif key == 'Normalize':
        retval = Normalize(*value)
    elif key == 'RandomizeBackground':
        retval = RandomizeBackground(*value)
    elif key == 'RandomizeBackgroundGraytone':
        retval = RandomizeBackgroundGraytone(*value)
    elif key == 'RandomizeBackgroundRGBNoise':
        retval = RandomizeBackgroundRGBNoise(*value)
    elif key == 'NotStupidRandomResizedCrop':
        retval = NotStupidRandomResizedCrop(*value)
    elif key == 'CoarseDropout':
        retval = CoarseDropout(*value)
    else:
        raise TypeError('key {} is not a recognized transformation class. ')
    return retval
