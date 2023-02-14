# eternal-bugshine

This repository houses methods related to feature visualization in models trained on beetles.

![](front_page/achenium_humile_0189_0_model_layers_layer3_4_num_iters_30_lr_0.1_smooth_coef_0_compressed.gif)

## Structure of the repository

- All core functionalities are stored in the `src` folder. 
  - `src/utils.py` stores modules implementing utilities for handling datasets, training models and visualizing results.
    - `src/utils/datasets.py` implements functionality for reading datasets, getting dataset statistics, splitting datasets and performing data augmentations.
    - `src/utils/headers.py` contains definitions of constants, in particular the imagenet dataset statistics and default parameters for the deepdream method
    - `src/utils/training.py` contains all functionality related to model training
    - `src/utils/visual.py` contains functions for visualizing tensors and plotting multiple system on the same graph
  - `src/deep_dream_aux.py` implements functionality used by the deepdream algorithm such as functions for calculating scale-space levels, smoothing gradients and converting to and from tensors
  - `src/deep_dream.py` implements functions for implementing deep dream scale space and gradient ascent
  - `src/models.py` contains classes exposing layers in different models, currently ResNet50 and GoogleNet.
- For a tutorial covering all functionality check `tutorial.ipynb`.
- To play with the deep dream algorithm use `playground.ipynb`.
- Note that the repo is a work in progress! This means there are most likely still many bugs present. 
  - currently you can only run the code from the two Jupyter notebooks.
  - GoogleNet does not support training from scratch
