# -*- coding:utf-8 -*-
import keras.backend as K


def MAE(y_true, y_pred):
    return K.abs(K.sum(y_true) - K.sum(y_pred))


def MSE(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))
