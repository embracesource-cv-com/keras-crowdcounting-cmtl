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
from utils.metrics import mae, mse
from utils.data_loader import DataLoader
from config import current_config as cfg
import os
import argparse
import tensorflow as tf
import keras


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    set_gpu_growth()
    dataset = args.dataset  # 'A' or 'B'
    cfg.init_path(dataset)  # 初始化路径名

    # 加载数据生成器
    train_data_gen = DataLoader(cfg.TRAIN_PATH,
                                cfg.TRAIN_GT_PATH,
                                shuffle=True)
    val_data_gen = DataLoader(cfg.VAL_PATH,
                              cfg.VAL_GT_PATH)

    class_weights = train_data_gen.get_class_weights()

    # 定义模型 & 编译
    input_shape = (None, None, 1)
    model = CMTL(input_shape)
    adam = Adam(lr=1e-5)
    loss = {'density': 'mse', 'cls': 'categorical_crossentropy'}
    loss_weights = {'density': 1.0, 'cls': 1e-4}
    print('[INFO] Compiling model ...'.format(dataset))
    model.compile(optimizer=adam, loss=loss, loss_weights=loss_weights,
                  metrics={'density': [mae, mse], 'cls': 'accuracy'})

    # 定义callback
    check_pointer = ModelCheckpoint(
        filepath=cfg.WEIGHT_PATH,
        monitor='loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='min'
    )
    callback_list = [check_pointer]

    # # 随机数据增广
    # print('[INFO] Random data augment ...'.format(dataset))
    # train_X, train_Y_den = train_data_loader.random_augment(train_X, train_Y_den)
    # 训练
    print('[INFO] Training Part_{} ...'.format(dataset))
    model.fit_generator(train_data_gen,
                        validation_data=val_data_gen,
                        epochs=cfg.EPOCHS,
                        callbacks=callback_list,
                        class_weight={"cls": class_weights},
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
