import typing as t
import torch
from torchvision import transforms
import numpy as np
from numpy import typing as npt

IMG_EXT= t.Literal['.png', '.jpg', '.pdf', '.gif']

class DreamConfig(t.TypedDict):
    """Contains parameters and settings used in the dreamspace function"""
    out_info:           t.Dict[str, t.Union[t.Tuple[int, int], t.List[int], int, None]]
    mean:               npt.NDArray[np.float32]
    std:                npt.NDArray[np.float32]
    input_img_path:     t.Optional[str]
    target_shape:       t.Union[int, t.Tuple[int, int]]

    noise:              t.Optional[t.Literal['uniform', 'gaussian']]
    correlation :       t.Optional[t.Literal['gaussian']]
    correlation_std :   float
    ratio:              float
    levels:             int
    shift_size:         int
    num_iters:          int
    lr:                 float
    loss_type:          t.Literal['norm', 'mean', 'MSE']
    loss_red:           t.Literal['mean', 'sum']
    norm_type:          t.Literal['standardize', 'abs_mean']
    eps:                float
    smooth:             bool
    kernel_size:        int
    smooth_coef:        float
    clamp_type:         t.Literal['standardize', 'unit', 'neg-unit']
    show:               bool
    figsize:            t.Tuple[int, int]
    save_interval:      int
    dpi:                int
    output_img_path:    t.Optional[str]
    img_overwrite:      bool
    video_path:         t.Optional[str]
    video_overwrite:    bool

class ModelConfig(t.TypedDict):
    """Holds information used when loading a model"""
    model_name:         str
    """Either the name of a stored model or None if using purely pretrained model"""
    model_architecture: t.Literal['resnet18', 'resnet34', 'resnet50']
    """Specify the architecture of the model"""
    pretrained:         bool
    """Decide if the loaded model should be pretrained or not"""
    device:             torch.device
    """Decide if model runs on cpu or gpu"""

class DatasetConfig(t.TypedDict):
    """Describes a dataset and the parameters of its dataloaders"""
    image_folder_path:          str
    """Path to the dataset images"""
    num_classes:                int
    """Number of classes in the dataset"""
    mean:                       npt.NDArray[np.float32]
    """Mean value of the dataset images"""
    std:                        npt.NDArray[np.float32]
    """Standard deviation of the dataset images"""
    average_image_shape:        t.Tuple[int, int]
    """The average image dimensions of the dataset"""
    data_augmentations:         t.List[t.Union[torch.nn.Module, object]]
    """List of data augmentations that should be applied to the training set"""
    batch_size:                 int
    """Batch size for the dataloaders"""
    num_workers:                int
    """Number of workers for the dataloaders"""
    rng_seed:                   int
    """RNG seed used to ensure reproducibility of training, validation and test sets"""
    training_data_ratio:        float
    """Specifies the percentwise size of the training set relative to the dataset"""
    validation_data_ratio:      float
    """Specifies the percentwise size of the validation set relative to the dataset"""

class OptimArgs(t.TypedDict):
    """Holds parameters used with the torch.optim.Adam optimizer"""
    lr:                         float
    eps:                        float
class EarlyStoppingArgs(t.TypedDict):
    """Holds parameters used to determine early stopping during training"""
    min_epochs:                 int
    patience:                   int
    min_delta:                  t.Union[int, float]

class TrainingInformation(t.TypedDict):
    """Holds information about a training session"""
    num_epochs:                 int
    trained_epochs:             int
    lr_decay:                   float
    stopped_early:              bool
    test_acc:                   float

class TrainingConfig(t.TypedDict):
    """Describes parameters used for training, besides model and dataset"""
    optim:                      t.Optional[torch.optim.Optimizer]
    optim_args:                 OptimArgs
    criterion:                  t.Optional[torch.nn.Module]
    scheduler:                  t.Optional[object]
    early_stopping:             t.Optional[object] 
    early_stopping_args:        EarlyStoppingArgs
    train_info:                 TrainingInformation

class PlotConfig(t.TypedDict):
    """Holds parameters used to plot during training"""
    size_h:                     int
    size_w:                     int
    fig_column:                 int
    fig_row:                    int

    show_title:                 bool
    titles:                     t.List[str]

    # at the moment we use the same name for label and y_label
    use_title_label:            bool
    y_label:                    t.Optional[t.List[str]]
    
    x_label:                    str
    label:                      t.List[str]
    show_grid:                  bool

    save_dpi:                   int
    save_figure:                bool
    save_subfigures:            bool
    save_padding:               float
    save_extension:             str   
    save_copy_png:              bool

    # these two might need to be some plt type instead
    param_linestyle:            str
    average_linestyle:          str
    param_alpha:                float

    rolling_avg_window:         int
    rolling_avg_label:          t.Optional[str]

    show_rolling_avg:           bool

ETERNAL_CONFIG = t.Union[PlotConfig, TrainingConfig, DatasetConfig, DreamConfig]