import os
from data_loader import load_images, load_merged_masks
from augmentation import data_augmentation
from visualization import show_mask
from multiresunet_model import build_multiresunet
from metrics_and_losses import dice_coefficient, precision, recall, iou_metric, combined_loss
from train_pipeline import train
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set image size and root directory
SIZE = 256
ROOT_PATH = "DATASET/"

# Load data paths
classes = sorted(os.listdir(ROOT_PATH))
mask_paths = []
image_paths = []

for cls in classes:
    mask_files = sorted([f for f in os.listdir(os.path.join(ROOT_PATH, cls)) if 'mask' in f])
    for mask_file in mask_files:
        mask_paths.append(os.path.join(ROOT_PATH, cls, mask_file))
        image_paths.append(os.path.join(ROOT_PATH, cls, mask_file.replace('_mask', '')))

# Load images and masks
images = load_images(image_paths, SIZE)
masks = load_merged_masks(image_paths, SIZE)

# Data sanity check
if images.size == 0 or masks.size == 0:
    raise ValueError("No images or masks loaded. Please check the directory structure.")

# Show some samples
plt.figure(figsize=(12, 8))
for i in range(15):
    idx = np.random.randint(len(images))
    plt.subplot(3, 5, i+1)
    show_mask(images[idx], masks[idx])
plt.tight_layout()
plt.show()

# Perform data augmentation
images, masks = data_augmentation(images, masks)
print(f"Augmented images shape: {images.shape}, Augmented masks shape: {masks.shape}")

# Build and compile the MultiResUNet model
input_shape = (SIZE, SIZE, 3)
model = build_multiresunet(input_shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient, precision, recall, iou_metric]
)

# Train the model
train(model, images, masks, epochs=50, batch_size=16)

# Save final model (optional)
model.save("final_multiresunet_model.h5")
print("Training complete, model saved.")
