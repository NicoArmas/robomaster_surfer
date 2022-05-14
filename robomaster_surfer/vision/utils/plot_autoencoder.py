import math

import torch
import wandb
from matplotlib import pyplot as plt


def get_shape(size):
    if size == 0 or size == 1:
        return 1, 1
    if size == 2:
        return 2, 1
    elif size == 4:
        return 2, 2
    elif size == 6:
        return 3, 2
    elif size == 8:
        return 4, 2
    elif size == 16:
        return 4, 4
    elif size == 32:
        return 8, 4
    elif size == 64:
        return 8, 8
    elif size == 128:
        return 16, 8


def plot_stages(autoencoder, data, save=False, plot=True, path=None):
    """
    It takes an autoencoder, a data point, and a few other parameters, and plots the input, the latent space, the decoded
    output, and the reconstruction error

    :param autoencoder: the autoencoder model
    :param data: the data to be plotted
    :param save: if True, saves the plot to the specified path, defaults to False (optional)
    :param plot: if True, the plot will be shown. If False, it will be saved, defaults to True (optional)
    :param path: the path to the dataset
    """
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(data[0].cpu().numpy())
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title('latent space')
    autoencoder.eval()
    with torch.no_grad():
        encoded, decoded = autoencoder(torch.unsqueeze(data, 0))
    new_shape = get_shape(encoded[0].shape[0])
    plt.imshow(encoded[0].cpu().numpy().reshape(new_shape))
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title('decoded')
    plt.imshow(decoded[0][0].cpu().numpy())
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Reconstruction error')
    plt.imshow(data[0].cpu().numpy() - decoded[0][0].cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
    plt.axis('off')

    if save:
        if path is None:
            raise ValueError('Name must be specified if save is True')
        plt.savefig(path)
        plt.close()
    if plot:
        plt.show()
