# -*- coding: utf-8 -*-
"""
   File Name:     metrics.py
   Description:   评估指标
   Author:        steven.yi
   date:          2019/04/17
"""
import keras.backend as K


def MAE(y_true, y_pred):
    return K.abs(K.sum(y_true) - K.sum(y_pred))


def MSE(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))
