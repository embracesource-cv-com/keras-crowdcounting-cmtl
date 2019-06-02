# -*- coding: utf-8 -*-
"""
   File Name：     losses.py
   Description :   损失函数
   Author :       mick.yi
   Date：          2019/5/30
"""
import tensorflow as tf
from keras import backend as K
import keras


def l2(y_true, y_pred):
    """
    均方差求和
    :param y_true: [B,H,W,1]
    :param y_pred: [B,H,W,1]
    :return:
    """
    shape = tf.shape(y_pred)
    b = shape[0]
    y_pred = tf.reshape(y_pred, [b, -1])
    y_true = tf.reshape(y_true, [b, -1])
    l2 = tf.sqrt(tf.reduce_sum(tf.square(y_pred-y_true), axis=-1))

    return l2
