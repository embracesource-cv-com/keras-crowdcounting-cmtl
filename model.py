# -*- coding: utf-8 -*-
from keras.models import Model, Input
from keras.layers import Conv2D, Dense, Activation, Concatenate, MaxPooling2D, Conv2DTranspose, Flatten
from keras.layers.advanced_activations import PReLU
from utils.spp import SpatialPyramidPooling


def CMTL(input_shape=None, num_classes=10):
    """
    定义模型
    :param input_shape:
    :param num_classes:
    :return:
    """
    inputs = Input(shape=input_shape)

    # shared layer
    shared = Conv2D(16, (9, 9), padding='same')(inputs)
    shared = PReLU(shared_axes=[1, 2])(shared)
    shared = Conv2D(32, (7, 7), padding='same')(shared)
    shared = PReLU(shared_axes=[1, 2])(shared)

    # high-level prior stage
    hl_prior_1 = Conv2D(16, (9, 9), padding='same')(shared)
    hl_prior_1 = PReLU(shared_axes=[1, 2])(hl_prior_1)
    hl_prior_1 = MaxPooling2D(2)(hl_prior_1)
    hl_prior_1 = Conv2D(32, (7, 7), padding='same')(hl_prior_1)
    hl_prior_1 = PReLU(shared_axes=[1, 2])(hl_prior_1)
    hl_prior_1 = MaxPooling2D(2)(hl_prior_1)
    hl_prior_1 = Conv2D(16, (7, 7), padding='same')(hl_prior_1)
    hl_prior_1 = PReLU(shared_axes=[1, 2])(hl_prior_1)
    hl_prior_1 = Conv2D(8, (7, 7), padding='same')(hl_prior_1)
    hl_prior_1 = PReLU(shared_axes=[1, 2])(hl_prior_1)

    # fix different sizes input to the same size output,
    # spp_out shape will be (samples, channels * sum([i * i for i in pool_list])
    spp_out = SpatialPyramidPooling([1, 2, 4])(hl_prior_1)
    hl_prior_2 = Dense(512)(spp_out)
    hl_prior_2 = PReLU(shared_axes=[1])(hl_prior_2)
    hl_prior_2 = Dense(256)(hl_prior_2)
    hl_prior_2 = PReLU(shared_axes=[1])(hl_prior_2)
    hl_prior_2 = Dense(num_classes)(hl_prior_2)
    hl_prior_2 = PReLU(shared_axes=[1])(hl_prior_2)
    cls = Activation('softmax', name='output_class')(hl_prior_2)

    # density estimate stage
    den_1 = Conv2D(20, (7, 7), padding='same')(shared)
    den_1 = PReLU(shared_axes=[1, 2])(den_1)
    den_1 = MaxPooling2D(2)(den_1)
    den_1 = Conv2D(40, (5, 5), padding='same')(den_1)
    den_1 = PReLU(shared_axes=[1, 2])(den_1)
    den_1 = MaxPooling2D(2)(den_1)
    den_1 = Conv2D(20, (5, 5), padding='same')(den_1)
    den_1 = PReLU(shared_axes=[1, 2])(den_1)
    den_1 = Conv2D(10, (5, 5), padding='same')(den_1)
    den_1 = PReLU(shared_axes=[1, 2])(den_1)
    merges = Concatenate(axis=-1)([hl_prior_1, den_1])
    den_2 = Conv2D(24, (3, 3), padding='same')(merges)
    den_2 = PReLU(shared_axes=[1, 2])(den_2)
    den_2 = Conv2D(32, (3, 3), padding='same')(den_2)
    den_2 = PReLU(shared_axes=[1, 2])(den_2)
    den_2 = Conv2DTranspose(16, (4, 4), strides=2, padding='same')(den_2)
    den_2 = PReLU(shared_axes=[1, 2])(den_2)
    den_2 = Conv2DTranspose(8, (4, 4), strides=2, padding='same')(den_2)
    den_2 = PReLU(shared_axes=[1, 2])(den_2)
    density_map = Conv2D(1, (1, 1), padding='same', activation='relu', name='output_density')(den_2)

    model = Model(inputs=inputs, outputs=[density_map, cls])
    return model
