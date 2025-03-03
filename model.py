import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from .loss import combined_loss
from .metrices import dice_coefficient,precision,recall,iou_metric,sensitivity

class SaveBestModel(Callback):
    def __init__(self, filepath):
        self.filepath, self.best_score = filepath, -np.inf
    def on_epoch_end(self, epoch, logs=None):
        score = logs['val_precision'] + logs['val_recall']
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.filepath)

def train(model, images, masks, epochs=50, batch_size=16):
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.15, random_state=42)
    model.compile(Adam(1e-3), combined_loss, metrics=['accuracy', dice_coefficient, precision, recall, iou_metric,sensitivity])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs,
              callbacks=[SaveBestModel('model.h5'), ReduceLROnPlateau()])
