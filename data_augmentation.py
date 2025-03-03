import numpy as np
import tensorflow as tf
import cv2

def data_augmentation(images, masks):
    augmented_images, augmented_masks = [], []
    for img, mask in zip(images, masks):
        for aug in range(5):
            img_aug, mask_aug = img, mask
            if aug == 1:
                img_aug = tf.image.adjust_contrast(img, 2)
            elif aug == 2:
                img_aug = tf.image.adjust_brightness(img, 0.3)
            elif aug == 3:
                img_aug, mask_aug = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
            elif aug == 4:
                img_aug = cv2.GaussianBlur(img, (5, 5), 0)
            augmented_images.append(img_aug)
            augmented_masks.append(mask_aug)
    return np.array(augmented_images), np.array(augmented_masks)
