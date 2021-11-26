from .custom_types import *
from .config import Config

import cv2 as cv
import numpy.fft as fft
from torchvision import transforms

# --------------- Base Classes ---------------
class Sharpen_class():
    """Base class for applying sharpening to image"""
    def __init__(self, config: Config, img: npt.NDArray[t.Any] = None):
        self.conv = Channel_convolution()
        self.scale_type = config.dream['scale_type']
        self.mean = config.mean
        self.std = config.std

    def apply(self, img: npt.NDArray[t.Any], level : int = None):
        return img

class Scale_class():
    """Base class for manipulating image scales"""
    def __init__(self, config: Config, sharpen_class : Sharpen_class = None, img: npt.NDArray[t.Any] = None):
        self.levels = config.dream['levels']
        self.mean = config.mean
        self.std = config.std
        self.clip_min = (0 - self.mean)/self.std
        self.clip_max = (1 - self.mean)/self.std
        self.ratio = config.dream['ratio']
        self.device = config.device

        if sharpen_class is None:
            self.sharpen_class = Sharpen_class(config, img)
        else:
            self.sharpen_class = sharpen_class(config, img)

    def get_first_level(self, img: npt.NDArray[t.Any]):
        """Returns starting level as a tensor"""
        ret = np.clip(img, self.clip_min, self.clip_max)
        return self.image_to_tensor(ret)

    def get_level(self, img: npt.NDArray[t.Any], index: int):
        if (index + 1) != self.levels:
            """Returns intermediate level as tensor"""
            ret = self.apply_scale(img, index)
            ret = self.apply_sharpen(ret, index)
            ret = np.clip(ret, self.clip_min, self.clip_max)
            return self.image_to_tensor(ret)
        return self.image_to_tensor(img)

    def apply_scale(self, img: npt.NDArray[t.Any], index: int):
        """Apply image scale"""
        return img

    def apply_sharpen(self, img: npt.NDArray[t.Any], index: int):
        return self.sharpen_class.apply(img, index)

    def image_to_tensor(self, img: npt.NDArray[t.Any]) -> torch.Tensor:
        """Convert output to tensor with grad"""
        tensor = transforms.ToTensor()(img).type(torch.float32).to(self.device).unsqueeze(0)
        tensor.requires_grad = True
        return tensor

class Channel_convolution():
    "Apply 2D convolution for each channel in the image"
    def conv(self, img : npt.NDArray[np.float32], kernel : npt.NDArray[np.float32] = None, sigma : float = 1, shift : bool = False):
        h,w,c = img.shape

        if kernel is None:
            kernel = self.gaussian_kernel(h, w, sigma)

        fft_kernel = fft.fft2(kernel, (h, w))
        ret = np.zeros(img.shape)

        for i in range(c):
            img_fft = fft.fft2(img[:,:,i].astype(np.float32))
            if shift:
                ret[:,:,i] = fft.fftshift(fft.ifft2(img_fft * fft_kernel).real)
            else:
                ret[:,:,i] = fft.ifft2(img_fft * fft_kernel).real

        return ret
    
    def gaussain(self, x : np.int32, y : np.int32, sigma):
        "2D gaussian function"
        exponent = ((x)**2 + (y)**2) / (2.0 * sigma **2)
        normalizer = 1.0 / (2.0 * np.pi * sigma **2)
        return normalizer * np.exp(-exponent)

    def gaussian_kernel(self, h : np.int32, w : np.int32, sigma : np.float32):
        "Creates a gaussian kernel of size (h, w)"
        y = np.linspace(-h//2, h//2-1, h)
        x = np.linspace(-w//2, w//2-1, w)
        X, Y = np.meshgrid(x, y)
        return self.gaussain(X, Y, sigma)


# --------------- Sharpening Classes ---------------

class DOG_sharpen(Sharpen_class):
    """Apply DOG sharpening to image pyramid or scale space"""
    def __init__(self, config: Config, img: npt.NDArray[t.Any]):
        super().__init__(config, img)
        self.ratio = config.dream['ratio']
        self.start_size = img.shape[:-1]
        self.levels = config.dream['levels']
        self.calc_dog_ratios(img)

    def apply(self, img: npt.NDArray[t.Any], level : int = None):
        ret = img*self.std + self.mean
        ret = ret * np.abs(self.dog_ratio[level] + 1)
        return (ret - self.mean) / self.std

    def calc_dog_ratios(self, img: npt.NDArray[t.Any]):
        """Precalculate DOG sharpening"""
        denorm_img = img*self.std + self.mean
        self.dog_ratio = []
        current_scale = denorm_img
            
        for level in range(self.levels-1):
            next_scale = self.scale_image(current_scale, index = (level+1))
            if self.scale_type == 'image_pyramid':
                rescale = self.scale_image(next_scale, index=current_scale.shape[:-1])
                self.dog_ratio.append(current_scale - rescale)
            else:
                self.dog_ratio.append(current_scale - next_scale)
            current_scale = next_scale
        self.dog_ratio.reverse()
         
    def scale_image(self, 
                    img: npt.NDArray[t.Any], 
                    index: t.Union[int, t.Tuple[int, int]]
                    ) -> npt.NDArray[t.Any]:
        if self.scale_type == 'image_pyramid':
            """Apply down or upscaling according to level"""
            if type(index) == int:
                h, w = np.round(np.float32(np.array(self.start_size)) *
                            (self.ratio ** (-index))).astype(np.int32)
            else:
                h, w = index
            return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR_EXACT)

        elif self.scale_type == 'scale_space':
            """Apply Gaussian smoothing according to nr of levels"""
            # no need to apply a gaussian with a sigma less than 1
            if index > 1:
                return np.clip(self.conv.conv(img, sigma = self.ratio * (index - 1), shift=True), 0, 1)
            return img

        else:
            raise RuntimeError('Unknown scale type')


class Laplace_sharpen(Sharpen_class):
    """Apply Laplace sharpening to image"""
    def __init__(self, config: Config, img: npt.NDArray[t.Any]):
        super().__init__(config, img)
        self.laplace = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        self.laplace_factor = config.dream['laplace_factor'] / config.dream['ratio']

    def apply(self, img: npt.NDArray[t.Any], level : int = None):
        laplace = self.conv.conv(img, self.laplace)

        return img + laplace/self.laplace_factor

# --------------- Scale Classes ---------------

class Image_pyramid(Scale_class):
    """Image pyramid as multiframe"""
    def __init__(self, config: Config, sharpen_class : Sharpen_class = None, img: npt.NDArray[t.Any] = None):
        super().__init__(config, sharpen_class, img)
        self.start_size = img.shape[:-1]

    def get_first_level(self, img: npt.NDArray[t.Any]):
        ret = self.apply_scale(img, -1)
        ret = np.clip(ret, self.clip_min, self.clip_max)
        return self.image_to_tensor(ret)

    def apply_scale(self, img: npt.NDArray[t.Any], index: int):
        """Apply down or upscaling according to level"""
        exponent = (index) - (self.levels - 2)
        h, w = np.round(np.float32(np.array(self.start_size)) *
                        (self.ratio ** exponent)).astype(np.int32)
        return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR_EXACT)

class Scale_space(Scale_class):
    """Scale space as multiframe"""
    def __init__(self, config: Config, sharpen_class : Sharpen_class = None, img: npt.NDArray[t.Any] = None):
        super().__init__(config, sharpen_class, img)
        self.conv = Channel_convolution()

    def get_first_level(self, img: npt.NDArray[t.Any]):

        sigma = (self.levels - 1) * self.ratio

        # no need to apply a gaussian with a sigma less than 1
        if sigma > 1:
            ret = self.conv.conv(img, sigma = sigma, shift=True)
        else:
            ret = img
        
        ret = np.clip(ret,self.clip_min,self.clip_max)
        return self.image_to_tensor(ret)
