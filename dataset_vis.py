import matplotlib.pyplot as plt
import tensorflow as tf

def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title:
        plt.title(title)
    plt.axis('off')

def show_mask(image, mask, cmap='jet', alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')
