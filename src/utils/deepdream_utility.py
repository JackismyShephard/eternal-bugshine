import cv2 as cv


from .custom_types import *
from .config import add_info_to_path, save, Config
from .visual import reshape_image, get_noise_image, save_video, save_img


def add_zeros(idx):
    ret = str(idx)
    if len(ret) == 1:
        return "00" + ret
    elif len(ret) == 2:
        return "0" + ret
    else:
        return ret

def get_start_image(config : Config, img : t.Optional[npt.NDArray[np.float32]] = None):
    if img is not None:
        input_img = img
    elif config.dream['input_img_path'] is not None:
        input_img = cv.imread(config.dream['input_img_path'])[:, :, ::-1]
        input_img = reshape_image(input_img, config.dream['target_shape'])
        input_img = input_img.astype(np.float32)  # convert from uint8 to float32
        input_img /= np.array(255.0)  # get to [0, 1] range
    elif config.dream['noise'] is not None:
        input_img = get_noise_image(config.dream['noise'], config.dream['target_shape'], 
                                config.dream['correlation'], config.dream['correlation_std'], 
                                config.dream['noise_scale'])
        input_img = input_img.astype(np.float32)
        input_img = (input_img - input_img.min())/(input_img.max() - input_img.min() )
    else:
        raise RuntimeError('img, input_img_path and noise are all None')
    return (input_img - config.dream['mean']) / config.dream['std']

def save_output(output_images : t.List[npt.NDArray[np.uint8]], config : Config):
    if config.dream['add_path_info']:
        path_info = config.model['model_name']
        for (layer, idxs) in config.dream['target_dict'].items():
            path_info =  path_info + '_' + layer + '_' + add_zeros(idxs)

    else:
        path_info = None
    if config.dream['output_img_path'] is not None:
        path = add_info_to_path(config.dream['output_img_path'], 
            path_info, config.dream['output_img_ext'], config.dream['img_overwrite'])
        if config.dream['save_meta']:
            save(path, config.model, config.dataset, config.training,  dream_config = config.dream)
        save_img(output_images[-1], path)

    if config.dream['video_path'] is not None:
        path = add_info_to_path(config.dream['video_path'], path_info, 
                                    config.dream['video_ext'], config.dream['video_overwrite'])
        if config.dream['save_meta']:
            save(path, config.model, config.dataset, config.training, dream_config = config.dream)
        save_video(path, output_images, config.dream['target_shape'])