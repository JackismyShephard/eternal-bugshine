import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils.headers import IMAGENET_MEAN, IMAGENET_STD

def show_tensor(inp, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                title=None, save_path=None, dpi=200, close=False,
                figsize=(7, 7), show_axis='on'):

    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(inp)
    plt.axis(show_axis)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        Path("figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight',
                    dpi=dpi, facecolor='w')
    if close:
        plt.close()
    plt.pause(0.001)  # pause a bit so that plots are updated


def multiplot(systems, x_axis, y_axis, labels, name=None,
              title=None, dpi=200):
    plt.figure()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    for i in range(len(systems)):
        plt.plot(systems[i, 0], systems[i, 1], label=labels[i])
        plt.title(title, pad=20)
    plt.legend()
    plt.grid()
    if name != None:
        Path("figures").mkdir(parents=True, exist_ok=True)
        plt.savefig('figures/'+name,  bbox_inches='tight',
                    facecolor='w', dpi=dpi)
    plt.show()
    plt.close()
