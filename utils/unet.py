from keras.layers import Conv2D, Activation
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                             padding="same", strides=(1, 1), instance_normalization=False):
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)

    if batch_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation("relu")(layer)
    else:
        return activation()(layer)


def create_dilate_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                                    padding="same", strides=(1, 1), instance_normalization=False, dilate_rate=1):
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides, dilation_rate=dilate_rate)(input_layer)

    if batch_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation("relu")(layer)
    else:
        return activation()(layer)