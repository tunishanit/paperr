import tensorflow as tf
from keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def iou_metric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1.) / (union + 1.)


def precision(y_true, y_pred):
    return K.sum(y_true * y_pred) / (K.sum(y_pred) + K.epsilon())

def recall(y_true, y_pred):
    return K.sum(y_true * y_pred) / (K.sum(y_true) + K.epsilon())
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    sensitivity_val = true_positives / (true_positives + false_negatives + K.epsilon())
    return sensitivity_val
