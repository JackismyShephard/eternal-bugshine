


#TODO figure out if rescaling leaves artifacts in output image
def scale_level(img: npt.NDArray[t.Any], start_size: t.Tuple, level: int,
                ratio: float = 1.8, levels: int = 4, 
                gauss_filter : t.Optional[t.Tuple[int, int, float, float]] = None, size : t.Optional[t.Tuple[int, int]] = None) -> npt.NDArray[t.Any]:

    exponent = level - levels + 1
    h, w = np.round(np.float32(np.array(start_size)) *
                    (ratio ** exponent)).astype(np.int32)
    if size is not None:
        h,w = size

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


def conv_per_channel(img : npt.NDArray[np.float32], kernel : npt.NDArray[np.float32], shift : bool = False):
    "Apply 2D convolution for each channel in the image"
    h,w,c = img.shape
    fft_kernel = fft.fft2(kernel, (h, w))
    ret = np.zeros(img.shape)

    for i in range(c):
        img_fft = fft.fft2(img[:,:,i].astype(np.float32))
        if shift:
            ret[:,:,i] = fft.fftshift(fft.ifft2(img_fft * fft_kernel).real)
        else:
            ret[:,:,i] = fft.ifft2(img_fft * fft_kernel).real

    return ret

# ----Scale Space---- 
 
def gaussain(x : np.int32, y : np.int32, sigma):
    "2D gaussian function"
    exponent = ((x)**2 + (y)**2) / (2.0 * sigma **2)
    normalizer = 1.0 / (2.0 * np.pi * sigma **2)
    return normalizer * np.exp(-exponent)

def gaussian_kernel(h : np.int32, w : np.int32, sigma : np.float32):
    "Creates a gaussian kernel of size (h, w)"
    y = np.linspace(-h//2, h//2-1, h)
    x = np.linspace(-w//2, w//2-1, w)
    X, Y = np.meshgrid(x, y)
    return gaussain(X, Y, sigma)

def scale_space(img : npt.NDArray[np.float32], level:int, dream_config : DreamConfig, model_clip: bool = True):
    "Applies gaussian smoothing to input image for each channel"

    sigma = dream_config['ratio'] * level

    clip_min = 0
    clip_max = 1
    if model_clip:
        clip_min = (- dream_config['mean'])/dream_config['std']
        clip_max = (1 - dream_config['mean'])/dream_config['std']

    # no need to apply a gaussian with a sigma less than 1
    if sigma < 1:
        return np.clip(img,clip_min,clip_max)

    h, w = img.shape[0:2]
    kernel = gaussian_kernel(h,w, sigma)

    ret = conv_per_channel(img, kernel, shift=True)
    
    return np.clip(ret,clip_min,clip_max)

# ----Scale Space---- 

def apply_laplacian(img : npt.NDArray[np.float32], dream_config : DreamConfig):
    "Apply laplacian sharpening to a tensor image"

    # get the image from the gpu and convert to numpy image
    current_img = tensor_to_image(img)

    laplace_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    laplace = conv_per_channel(current_img, laplace_kernel)

    clip_min = (- dream_config['mean'])/dream_config['std']
    clip_max = (1 - dream_config['mean'])/dream_config['std']

    if dream_config['laplace_factor'] > 0:
        factor = dream_config['laplace_factor'] / dream_config['ratio']
    else:
        factor = 1

    return np.clip(current_img + laplace/factor, clip_min, clip_max)

def apply_DOG(img: torch.Tensor, original_img : npt.NDArray[np.float32], level : int, dream_config : DreamConfig):
    current_img = np.clip(tensor_to_image(img)* dream_config['std'] + dream_config['mean'],0,1)
    denormalized_img = np.clip(original_img * dream_config['std'] + dream_config['mean'], 0,1)

    difference = 0
    if dream_config["scale_type"] == "image_pyramid":
        start_size = original_img.shape[:-1]
        current_size = current_img.shape[:-1]
        
        img_downscale = scale_level(denormalized_img, start_size, level-1,
                                        dream_config['ratio'], dream_config['levels'])

        img_downscale = scale_level(img_downscale, start_size, level,
                                        dream_config['ratio'], dream_config['levels'], size = current_size)

        img_current_scale = scale_level(denormalized_img, start_size, level,
                                        dream_config['ratio'], dream_config['levels'], size = current_size)

        difference = img_current_scale - img_downscale

    if dream_config["scale_type"] == "scale_space":
        low_scale = scale_space(denormalized_img, dream_config['levels'] - level, dream_config, False)
        high_scale = scale_space(denormalized_img, dream_config['levels'] - (level+1), dream_config, False)

        difference = high_scale - low_scale

    ratio = np.abs(difference + 1)

    return  ((current_img * ratio)-dream_config['mean'])/dream_config['std']                                 
    
