from itertools import product
from typing import Callable

import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader

def get_fashion_mnist_subset(n_train_samples: int, n_val_samples: int, transforms: Callable) -> tuple[DataLoader, DataLoader]:
    full_train_set = FashionMNIST("sample_data", train=True, download=True, transform=transforms)
    train_set, val_set, _ = random_split(full_train_set,
        [n_train_samples, n_val_samples, int(6e4) - (n_train_samples + n_val_samples)])

    test_set = FashionMNIST("sample_data", train=False, download=True, transform=transforms)

    return train_set, val_set, test_set


def visualize_first_4(dataloader):
    for batch, labels in dataloader:
        break

    print(f"Shape of images is {batch.shape}")
    fig, axs = plt.subplots(2, 2)
    for i, j in product((0, 1), (0, 1)):
        axs[i, j].imshow(batch[2*i + j][0])
        axs[i, j].set_title(f"Label: {labels[2*i + j]}")
        axs[i, j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.show()


def plot_train_and_val_loss(train_l, val_l, title=""):
    plt.clf()
    plt.plot(train_l, label="Train loss")
    plt.plot(val_l, label="Validation loss")
    plt.axhline(y=min(val_l), c="r", linestyle="--")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.show()