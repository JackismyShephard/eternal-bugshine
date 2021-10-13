import typing as t
import torch
import numpy as np
from numpy import typing as npt
#IMPLEMENT custom types / type annotations for codebase
class DreamConfig(t.TypedDict, total=False):
    model:              torch.nn.Module
    model_aux_dict:     t.Optional[t.Dict[str, t.Any]]
    out_info:           t.Dict[str, t.Union[t.Tuple[int, int], int, None]]
    mean:               npt.NDArray[np.float32]
    std:                npt.NDArray[np.float32]
    input_img_path:     t.Optional[str]
    target_shape:       t.Union[int, t.Tuple[int, int]]
    noise:              t.Optional[t.Literal[   'uniform', 
                                                'gaussian', 
                                                'correlated_uniform', 
                                                'correlated_gaussian']]
    ratio:              float
    levels:             int
    shift_size:         int
    num_iters:          int
    lr:                 float
    loss_type:          t.Literal['norm', 'mean']
    loss_red:           t.Literal['mean', 'sum']
    norm_type:          t.Optional[t.Literal['standardize']] #TODO other one uses epsilon, what is it called?
    eps:                float
    smooth:             bool
    kernel_size:        int
    smooth_coef:        float
    clamp_type:         t.Optional[t.Literal['standardize', 'unit']]
    show:               bool
    figsize:            t.Tuple[int, int]
    save_interval:      int
    dpi:                int
    output_img_path:    t.Optional[str]
    img_overwrite:      bool
    video_path:         t.Optional[str]
    video_overwrite:    bool
    device:             t.Optional[torch.device]