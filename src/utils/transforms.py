from PIL import Image
import typing as t
import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import functional

# GENERAL TODOS
    #consider making general class for background substitution which can then be subclassed

    #consider making RGBtoGRAY into a function or use another built int function from skimage for this purpose

        #consider also using general noise function

    #consider making general wrapper class or utility function for transformations allowing the kind of representation we want

class RandomizeBackground(torch.nn.Module):  # TODO consider changing name to uniform_randomize_background or similar
    """Replace beetle PIL image background color with a random color."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, img: t.Union[Image.Image, torch.Tensor]):
        if not isinstance(img, Image.Image): # QUESTION consider also allowing uint numpy arrs in range 0 to 255?
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255

        np_x_gray = (np.sum(np_x, axis=2)) / 3 # QUESTION consider other ways than averaging of performing grayscale conversion?
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
    # TODO consider letting __repr__ just handle aggregating parameters into a list 
    # and making a corresponding dict with class name as key and list as val.
    # You can then (maybe) define __str_ to stringify this dict if necessary for json serialization.
    # But i dont understand why you have to make the dict a string in order to serialize to json. 
    # Couldnt you just save the dict when initing the class and then retrieving it during serialization?
    def __repr__(self):
        args = '[{}]'.format(self.cutoff)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'


class NotStupidRandomResizedCrop(torch.nn.Module):
    """A RandomResizedCrop reimplementation that does just what we need it to do.
        Crops a section of a PIL image with shape d*img.shape, where
        min_scale/100 <= d <= max_scale/100, at some random coordinate in the image."""

    def __init__(self, min_scale: float = 0.5, max_scale: float = 1):
        # TODO consider making final output size  an input parameter
        super().__init__()
        self.rng = np.random.default_rng() 
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.int_min_scale = int(min_scale * 100)
        self.int_max_scale = int(max_scale * 100)
    def forward(self, img: t.Union[Image.Image, torch.Tensor]):
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
        # TODO if we always downsample then we should use InterpolationMode.NEAREST or InterpolationMode.INTER_AREA
        img = functional.resize(img, [h,w]) 
        return img
    def __repr__(self):
        args = '[{},{}]'.format(self.min_scale, self.max_scale)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomizeBackgroundGraytone(torch.nn.Module):
    """Replace beetle PIL image background color with a random graytone."""
    def __init__(self, cutoff: float, min: float = 0, max: float = 1): 
        super().__init__()
        self.rng = np.random.default_rng()
        self.cutoff = cutoff
        self.min = min
        self.max = max
    def forward(self, img: t.Union[Image.Image, torch.Tensor]):
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        color = self.rng.integers(int(self.min * 255), int(self.max * 255))
        np_x = np.where(mask == True, color, (np_x * 255).astype('uint8')) # QUESTION why not just use img as third argument?
        return Image.fromarray(np_x)
    def __repr__(self):
        args = '[{},{},{}]'.format(self.cutoff, self.min, self.max)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomizeBackgroundRGBNoise(torch.nn.Module):
    """Replace beetle PIL image background color with RGB noise."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.rng = np.random.default_rng()
        self.cutoff = cutoff    
    def forward(self, img: t.Union[Image.Image, torch.Tensor]):
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img) / 255
        np_x_gray = (np.sum(np_x, axis=2)) / 3
        mask = np_x_gray > self.cutoff
        mask = np.dstack([mask, mask, mask])
        new_bg = self.rng.random(np_x.shape) # QUESTION consider using other distributions?
        np_x = np.where(mask == True, new_bg, np_x)
        np_x = (np_x * 255).astype('uint8')
        return Image.fromarray(np_x)
    def __repr__(self):
        args = '[{}]'.format(self.cutoff)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'
#TODO allow holes to be filled with noise or perhaps solid colors?
#TODO consider a more compact representation of class arguments
class CoarseDropout(torch.nn.Module):
    def __init__(self,  min_holes: int = 0, max_holes:int = 10, 
                        min_height: int = 5, max_height: int = 10, 
                        min_width:int = 5, max_width: int = 10):
        super().__init__()
        self.rng = np.random.default_rng()
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width

    def forward(self, img: t.Union[Image.Image, torch.Tensor]):
        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL.Image.Image. Got {}".format(type(img)))

        np_x = np.array(img)
        (h, w, _) = np_x.shape
        mask = np.ones(np_x.shape)
        holes = self.rng.integers(self.min_holes, self.max_holes)
        for _ in range(holes):
            width = self.rng.integers(self.min_width, self.max_width)
            height = self.rng.integers(self.min_height, self.max_height)
            x = self.rng.integers(0, w)
            y = self.rng.integers(0, h)
            mask[y:y+height,x:x+width,:] = 0
        np_x = (mask * np_x).astype('uint8')
        return Image.fromarray(np_x)
    def __repr__(self):
        args = '[{},{},{},{},{},{}]'.format(self.min_holes, self.max_holes, self.min_height, self.max_height, self.min_width, self.max_width)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

#wrapper classes to load json representations of torchvision transforms. probably exists a smarter way, but im doing it
class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.transform = torchvision.transforms.RandomVerticalFlip(p)
        self.p = p
    def forward(self, x):
        return self.transform(x)
    def __repr__(self):
        args = '[{}]'.format(self.p)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class RandomRotation(torch.nn.Module):
    def __init__(self, degrees=(0,0), fill=255):
        super().__init__()
        self.transform = torchvision.transforms.RandomRotation(degrees, functional.InterpolationMode.NEAREST, False, None, fill, None)
        self.degrees = degrees
        self.fill = fill
    def forward(self, x):
        return self.transform(x)
    def __repr__(self):
        args = '[[{},{}],{}]'.format(self.degrees[0], self.degrees[1], self.fill)
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class Resize(torch.nn.Module):
    def __init__(self, size=(200,200)):
        super().__init__()
        self.transform = torchvision.transforms.Resize(size, functional.InterpolationMode.BILINEAR, None, None)
        self.size = size
    def forward(self, x):
        return self.transform(x)
    def __repr__(self):
        args = '[[{},{}]]'.format(self.size[0], self.size[1])
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

# QUESTION why is this not a torch module?
class ToTensor:
    def __init__(self):
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
    def __call__(self, x):
        return functional.to_tensor(x)
    def __repr__(self):
        args = '[]'
        return '{"'+self.__class__.__name__ +'":'+'{}'.format(args) + '}'

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.transform = torchvision.transforms.Normalize(mean, std, inplace)
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def forward(self, x):
        return self.transform(x)
    def __repr__(self):
        args = '[{}, {}]'.format(str(list(self.mean)), str(list(self.std)))
        return '{'+'"{}":'.format(self.__class__.__name__) +''+'{}'.format(args) + '}'

#TODO make transform dict a custom type
# QUESTION currently only the output value of the last transformation
# in transform_dict is returned. what exactly is the purpose of this function?
def string_to_class(transform_dict: dict):
    for key, value in transform_dict.items():
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
