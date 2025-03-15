import matplotlib.pyplot as plt
from itertools import product

def visualize_first_4(batch, labels):
    fig, axs = plt.subplots(2, 2)
    for i, j in product((0, 1), (0, 1)):
        axs[i, j].imshow(batch[2*i + j][0])
        axs[i, j].set_title(f"Label: {labels[2*i + j]}")
        axs[i, j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.show()