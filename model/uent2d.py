from keras.layers import Conv2D, Concatenate, LeakyReLU, Add, Input, Activation, UpSampling2D, SpatialDropout2D, \
    MaxPooling2D, Dropout, concatenate
from keras.optimizers import Adam
from keras import Model
from functools import partial
from utils import create_convolution_block, weighted_dice_coefficient_loss, create_dilate_convolution_block


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)
create_dilate_convolution_block = partial(create_dilate_convolution_block, activation=LeakyReLU, instance_normalization=True)


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
    return convolution2


def create_1x1_convolution_module(input_layer, n_filters):
    filters = [n_filters*3, n_filters*2, n_filters]
    convolution = Conv2D(filters[0], kernel_size=(1, 1), padding="same")(input_layer)
    for filter in filters[1:]:
        convolution = Conv2D(filter, kernel_size=(1, 1), padding="same")(convolution)
    return convolution


def uent2d_model(input_shape=(256, 256, 1), n_base_fileters=16, depth=5, dropout_rate=0.3,
                 n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                 loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_fileters: 第一个卷积块的滤波器个数
    :param depth:  unet结构的深度
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    "原始的unet"
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()  # 每个卷积块的滤波器个数

    for level_num in range(depth):  # 遍历深度
        # 计算得到各卷积块的滤波器个数
        n_level_filters = (2**level_num)*n_base_fileters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2))

        # 创建本层的卷积块
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        # 创建卷积块的残差并且加入到输出列表,多尺度融合采用的方式是相加而不是concatenate能节约内存
        summation_layer = Add()([in_conv, context_output_layer])    # 这里实质上是实现本次的残差块
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # 下面进行上采样部分,从最下层开始：current_layer
    segmentation_layers = list()
    for level_num in range(depth-2, -1, -1):    # 3, 2, 1, 0
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_num])    # 先上采样，再卷积一次
        concatenation_layer = Concatenate(axis=-1)([level_output_layers[level_num], up_sampling])   # 当前水平层的通道上的skip connect
        localization_output = create_localization_module(concatenation_layer, level_filters[level_num])     # 对skip connect之后的结果进行两次卷积操作
        current_layer = localization_output
        if level_num < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_num in reversed(range(n_segmentation_levels)):  # 0, 1, 2--->2, 1, 0
        segmentation_layer = segmentation_layers[level_num]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_num > 0:   # 2, 1
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def uent2d_dilate_model(input_shape=(256, 256, 1), n_base_fileters=16, depth=5, dropout_rate=0.3,
                        n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                        loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_fileters: 第一个卷积块的滤波器个数
    :param depth:  unet结构的深度
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    "在buttonblock添加三个并联的空洞卷积率分别为2， 3， 5的分支，结果和buttonblock的输出执行sum操作，再进行后继解码操作"
    dilate_rate = [2, 3, 5]
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()  # 每个卷积块的滤波器个数

    for level_num in range(depth):  # 遍历深度
        # 计算得到各卷积块的滤波器个数
        n_level_filters = (2**level_num)*n_base_fileters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2))

        # 创建本层的卷积块
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        # 创建卷积块的残差并且加入到输出列表,多尺度融合采用的方式是相加而不是concatenate能节约内存
        summation_layer = Add()([in_conv, context_output_layer])    # 这里实质上是实现本次的残差块
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # 在此处添加空洞卷积操作，即在buttonblock处添加dilation
    filters = level_filters[-1]
    dilate_features = list()
    dilate_features.append(current_layer)
    for rate in dilate_rate:
        dilate_feature = create_dilate_convolution_block(current_layer, filters, dilate_rate=rate)
        feature = create_localization_module(dilate_feature, filters)
        dilate_features.append(feature)

    current_layer = Add()(dilate_features)

    # 下面进行上采样部分,从最下层开始：current_layer
    segmentation_layers = list()
    for level_num in range(depth-2, -1, -1):    # 3, 2, 1, 0
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_num])
        # 先上采样，再卷积一次
        concatenation_layer = Concatenate(axis=-1)([level_output_layers[level_num], up_sampling])
        # 当前水平层的通道上的skip connect
        localization_output = create_localization_module(concatenation_layer, level_filters[level_num])
        # 对skip connect之后的结果进行两次卷积操作
        current_layer = localization_output
        if level_num < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_num in reversed(range(n_segmentation_levels)):  # 0, 1, 2--->2, 1, 0
        segmentation_layer = segmentation_layers[level_num]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_num > 0:   # 2, 1
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def uent2d_dilate_model_1X1(input_shape=(256, 256, 1), n_base_fileters=16, depth=5, dropout_rate=0.3,
                            n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                            loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_fileters: 第一个卷积块的滤波器个数
    :param depth:  unet结构的深度
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    # 在buttonblock添加三个并联的空洞卷积率分别为2， 3， 5的分支，结果和buttonblock的输出执行concat操作
    # 执行完concat操作之后，使用多层1*1*C的卷积核进行特征融合增加非线性，
    # 注意，此处三个并联的空洞卷积的输出可以尝试分别使用sum和concat
    # 先concat之后再使用1*1卷积核效果会好一些？

    # 直接将4个分支进行concat，然后使用1*1卷积融合
    dilate_rate = [2, 3, 5]
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()  # 每个卷积块的滤波器个数

    for level_num in range(depth):  # 遍历深度
        # 计算得到各卷积块的滤波器个数
        n_level_filters = (2**level_num)*n_base_fileters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2))

        # 创建本层的卷积块
        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        # 创建卷积块的残差并且加入到输出列表,多尺度融合采用的方式是相加而不是concatenate能节约内存
        summation_layer = Add()([in_conv, context_output_layer])    # 这里实质上是实现本次的残差块
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # 在此处添加空洞卷积操作，即在buttonblock处添加dilation
    filters = level_filters[-1]
    dilate_features = list()
    dilate_features.append(current_layer)
    for rate in dilate_rate:
        dilate_feature = create_dilate_convolution_block(current_layer, filters, dilate_rate=rate)
        feature = create_localization_module(dilate_feature, filters)
        dilate_features.append(feature)

    # current_layer = Add()(dilate_features)
    merged_feature = Concatenate()(dilate_features)  # (?, 32, 32, 1024)
    current_layer = create_1x1_convolution_module(merged_feature, filters)

    # 下面进行上采样部分,从最下层开始：current_layer
    segmentation_layers = list()
    for level_num in range(depth-2, -1, -1):    # 3, 2, 1, 0
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_num])
        # 先上采样，再卷积一次
        concatenation_layer = Concatenate(axis=-1)([level_output_layers[level_num], up_sampling])
        # 当前水平层的通道上的skip connect
        localization_output = create_localization_module(concatenation_layer, level_filters[level_num])
        # 对skip connect之后的结果进行两次卷积操作
        current_layer = localization_output
        if level_num < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_num in reversed(range(n_segmentation_levels)):  # 0, 1, 2--->2, 1, 0
        segmentation_layer = segmentation_layers[level_num]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_num > 0:   # 2, 1
            output_layer = UpSampling2D(size=(2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def unet(pretrained_weights=None, input_size=(256, 256, 1), loss_function='binary_crossentropy'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = uent2d_model()
    model.summary()