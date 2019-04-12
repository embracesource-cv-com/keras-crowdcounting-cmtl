# -*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import CMTL
from utils.metrics import MAE, MSE
from utils.data_loader import DataLoader
import config as cfg
import os
import argparse


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = args.dataset  # 'A' or 'B'

    train_path = cfg.TRAIN_PATH.format(dataset)
    train_gt_path = cfg.TRAIN_GT_PATH.format(dataset)
    val_path = cfg.VAL_PATH.format(dataset)
    val_gt_path = cfg.VAL_GT_PATH.format(dataset)

    train_data_loader = DataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False)
    val_data_loader = DataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=False)
    # 加载数据
    print('Loading data, wait a moment...')
    train_X, train_Y_den, train_Y_class = train_data_loader.load_all()
    val_X, val_Y_den, val_Y_class = val_data_loader.load_all()
    class_weights = train_data_loader.get_class_weights()

    # 定义模型 & 编译
    input_shape = (None, None, 1)
    model = CMTL(input_shape)
    adam = Adam(lr=0.00001)
    loss = {'output_density': 'mse', 'output_class': 'categorical_crossentropy'}
    loss_weights = {'output_density': 1.0, 'output_class': 0.0001}
    model.compile(optimizer=adam, loss=loss, loss_weights=loss_weights,
                  metrics={'output_density': [MAE, MSE]})

    # 定义callback
    checkpointer_best_train = ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'mcnn_' + dataset + '_train.hdf5'),
        monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    callback_list = [checkpointer_best_train]

    # 训练
    model.fit(train_X, {"output_density": train_Y_den, "output_class": train_Y_class},
              validation_data=(val_X, {"output_density": val_Y_den, "output_class": val_Y_class}),
              batch_size=1, epochs=cfg.EPOCHS, callbacks=callback_list)
