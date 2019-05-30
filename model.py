# -*- coding: utf-8 -*-
"""
   File Name:     model.py
   Description:   定义模型
   Author:        steven.yi
   date:          2019/04/17
"""
from keras.models import Model, Input
from keras.layers import Conv2D, Dense, Activation, Concatenate, MaxPooling2D, Conv2DTranspose, Dropout, Lambda, Flatten
from keras.layers.advanced_activations import PReLU
import tensorflow as tf


def _conv_unit(input_tensor, filters, kernel_size, name):
    """
    卷积单元，包括一个卷积核一个PReLU激活
    :param input_tensor
    :param filters:
    :param kernel_size:
    :param name:
    :return:
    """
    x = Conv2D(filters, kernel_size, padding='same', name='conv_{}'.format(name))(input_tensor)
    x = PReLU(shared_axes=[1, 2], name="prelu_{}".format(name))(x)
    return x


def _fc_unit(input_tenser, units, name, with_drops=True):
    """
    一个全连接单元
    :param input_tenser:
    :param units:
    :param name:
    :return:
    """
    x = Dense(units, name='fc_{}'.format(name))(input_tenser)
    x = PReLU(shared_axes=[1], name='prelu_{}'.format(name))(x)
    if with_drops:
        x = Dropout(0.5)(x)
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
    x = _conv_unit(inputs, 16, 9, 'base_1')
    x = _conv_unit(x, 32, 7, 'base_2')
    shared = x

    # high-level prior stage
    x = _conv_unit(x, 16, 9, name='prior_1')
    x = MaxPooling2D(2)(x)
    x = _conv_unit(x, 32, 7, name='prior_2')
    x = MaxPooling2D(2)(x)
    x = _conv_unit(x, 16, 7, name='prior_3')
    x = _conv_unit(x, 8, 7, name='prior_4')
    prior = x

    # 空间池化，通道变为4
    x = Lambda(lambda c: tf.image.resize_images(c, [64, 64]))(prior)
    x = MaxPooling2D(2)(x)
    x = _conv_unit(x, 4, 1, name='prior_5')
    x = Flatten()(x)

    x = _fc_unit(x, 512, 'pre_cls_1')
    x = _fc_unit(x, 256, 'pre_cls_2')
    x = _fc_unit(x, num_classes, 'pre_cls_3', with_drops=False)
    cls = Activation('softmax', name='cls')(x)

    # density estimate stage
    x = _conv_unit(shared, 20, 7, name='dens_1')
    x = MaxPooling2D(2)(x)
    x = _conv_unit(x, 40, 5, name='dens_2')
    x = MaxPooling2D(2)(x)
    x = _conv_unit(x, 20, 5, name='dens_3')
    x = _conv_unit(x, 10, 5, name='dens_4')
    # 合并先验
    merges = Concatenate(axis=-1)([prior, x])
    x = _conv_unit(merges, 24, 3, 'dens_5')
    x = _conv_unit(x, 32, 3, 'dens_6')
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='conv_dens_7')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu_dens_7')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='conv_dens_8')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu_dens_8')(x)
    density_map = Conv2D(1, (1, 1), padding='same', activation='relu', name='density')(x)
    model = Model(inputs=inputs, outputs=[density_map, cls])
    return model


def main():
    m = CMTL(input_shape=(200, 200, 1), num_classes=10)
    m.summary()


if __name__ == '__main__':
    main()
