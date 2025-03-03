import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.image as tfi
import glob

def load_image(image, size):
    img = img_to_array(load_img(image))
    img = tfi.resize(img / 255., (size, size))
    return np.round(img, 4).astype(np.float32)

def load_images(image_paths, size, mask=False, trim=None):
    if trim:
        image_paths = image_paths[:trim]
    images = np.zeros((len(image_paths), size, size, 1 if mask else 3), dtype=np.float32)
    for i, path in enumerate(image_paths):
        img = load_image(path, size)
        images[i, :, :, 0] if mask else images[i] = img
    return images

def load_merged_masks(image_paths, size):
    merged_masks = np.zeros((len(image_paths), size, size, 1), dtype=np.float32)
    for i, path in enumerate(image_paths):
        masks = glob.glob(path.replace('.png', '*mask*.png'))
        merged_mask = np.zeros((size, size), dtype=np.float32)
        for mask_path in masks:
            mask = load_image(mask_path, size)
            merged_mask = np.logical_or(merged_mask, mask[:, :, 0])
        merged_masks[i] = np.expand_dims(merged_mask, -1)
    return merged_masks
