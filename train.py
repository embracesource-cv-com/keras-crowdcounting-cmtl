# -*- coding: utf-8 -*-
"""
   File Name:     train.py
   Description:   训练
   Author:        steven.yi
   date:          2019/04/17
"""
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import CMTL
from utils.metrics import MAE, MSE
from utils.data_loader import DataLoader
from config import current_config as cfg
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
    train_X, train_Y_den, train_Y_class = train_data_loader.load_all()
    val_X, val_Y_den, val_Y_class = val_data_loader.load_all()
    class_weights = train_data_loader.get_class_weights()

    # 定义模型 & 编译
    input_shape = (None, None, 1)
    model = CMTL(input_shape)
    adam = Adam(lr=0.00001)
    loss = {'density': 'mse', 'cls': 'binary_crossentropy'}
    loss_weights = {'density': 1.0, 'cls': 0.0001}
    print('[INFO] Compiling model ...'.format(dataset))
    model.compile(optimizer=adam, loss=loss, loss_weights=loss_weights,
                  metrics={'density': [MAE, MSE]})

    # 定义callback
    checkpointer_best_train = ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'mcnn_' + dataset + '_train.hdf5'),
        monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    callback_list = [checkpointer_best_train]

    # 随机数据增广
    print('[INFO] Random data augment ...'.format(dataset))
    train_X, train_Y_den = train_data_loader.random_augment(train_X, train_Y_den)
    # 训练
    print('[INFO] Training Part_{} ...'.format(dataset))
    model.fit(train_X,
              {"density": train_Y_den, "cls": train_Y_class},
              validation_data=(val_X, {"density": val_Y_den, "cls": val_Y_class}),
              batch_size=1, epochs=cfg.EPOCHS, callbacks=callback_list,
              class_weight={"cls": class_weights})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
