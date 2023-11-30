from matplotlib import pyplot as plt
from math import ceil

def save_grid_with_captions(images, captions, save_path, ncols):
    fig, axes = plt.subplots(ceil(len(images) / ncols), ncols, constrained_layout=True)

    for i in range(len(images)):
        ax = axes.ravel()[i]
        ax.imshow(images[i])
        ax.set_title(captions[i])

    fig.savefig(save_path)