# -*- coding: utf-8 -*-
from keras.models import Model, Input
from keras.layers import Conv2D, Dense, Activation, Concatenate, MaxPooling2D, Conv2DTranspose, Flatten
from keras.layers.advanced_activations import PReLU


def spp(x):
    return x


def CMTL(input_shape=None, num_classes=10):
    """
    定义模型
    :param input_shape:
    :param num_classes:
    :return:
    """
    inputs = Input(shape=input_shape)

    # shared layer
    shared = Conv2D(16, (9, 9), padding='same', activation=PReLU())(inputs)
    shared = Conv2D(32, (7, 7), padding='same', activation=PReLU())(shared)

    # high-level prior stage
    hl_prior_1 = Conv2D(16, (9, 9), padding='same', activation=PReLU())(shared)
    hl_prior_1 = MaxPooling2D(2)(hl_prior_1)
    hl_prior_1 = Conv2D(32, (7, 7), padding='same', activation=PReLU())(hl_prior_1)
    hl_prior_1 = MaxPooling2D(2)(hl_prior_1)
    hl_prior_1 = Conv2D(16, (7, 7), padding='same', activation=PReLU())(hl_prior_1)
    hl_prior_1 = Conv2D(8, (7, 7), padding='same', activation=PReLU())(hl_prior_1)
    spp_out = spp(hl_prior_1)  # todo: spp layer
    hl_prior_2 = Flatten()(spp_out)
    hl_prior_2 = Dense(512, activation=PReLU())(hl_prior_2)
    hl_prior_2 = Dense(256, activation=PReLU())(hl_prior_2)
    hl_prior_2 = Dense(num_classes, activation=PReLU())(hl_prior_2)
    cls = Activation('softmax', name='output_class')(hl_prior_2)

    # density estimate stage
    den_1 = Conv2D(20, (7, 7), padding='same', activation=PReLU())(shared)
    den_1 = MaxPooling2D(2)(den_1)
    den_1 = Conv2D(40, (5, 5), padding='same', activation=PReLU())(den_1)
    den_1 = MaxPooling2D(2)(den_1)
    den_1 = Conv2D(20, (5, 5), padding='same', activation=PReLU())(den_1)
    den_1 = Conv2D(10, (5, 5), padding='same', activation=PReLU())(den_1)
    merges = Concatenate(axis=-1)([hl_prior_1, den_1])
    den_2 = Conv2D(24, (3, 3), padding='same', activation=PReLU())(merges)
    den_2 = Conv2D(32, (3, 3), padding='same', activation=PReLU())(den_2)
    den_2 = Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation=PReLU())(den_2)
    den_2 = Conv2DTranspose(8, (4, 4), strides=2, padding='same', activation=PReLU())(den_2)
    density_map = Conv2D(1, (1, 1), padding='same', activation='relu', name='output_density')(den_2)

    model = Model(inputs=inputs, outputs=[density_map, cls])
    return model
