import keras.backend as K
import keras

def weighted_dice_coefficient(y_true, y_pred, axis=(1, 2), smooth=0.00001):
    return K.mean(2.*(K.sum(y_true*y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1 - weighted_dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    return (2. * intersection + smooth)/(K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def mix_loss(y_true, y_pred):

    return weighted_dice_coefficient_loss(y_true, y_pred) + \
           keras.losses.binary_crossentropy(y_true, y_pred, 0.00001)