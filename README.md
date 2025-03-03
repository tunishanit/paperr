# paperr

📁 Project Structure

#project_folder/
├── main.py                    # Main script to train the model
├── data_loader.py              # Handles loading images and masks
├── augmentation.py             # Applies image and mask augmentations
├── visualization.py            # Contains functions for visualizing images and masks
├── multiresunet_model.py       # Defines the MultiResUNet architecture
├── metrics_and_losses.py       # Defines custom metrics and losses
├── train_pipeline.py           # Manages the training pipeline
├── README.md                    # This documentation file
└── BUSBRA/                      # Folder containing the dataset (images + masks)



---

## 💻 Installation

Install necessary dependencies:

```bash
pip install tensorflow keras matplotlib scikit-learn opencv-python pillow



#Run
python main.py

