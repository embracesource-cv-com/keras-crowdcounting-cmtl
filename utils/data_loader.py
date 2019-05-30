# -*- coding:utf-8 -*-
"""
   File Name:     data_loader.py
   Description:   数据加载
   Author:        steven.yi
   date:          2019/04/17
"""
import numpy as np
import pandas as pd
import cv2
import os
import sys
from keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self, data_path, gt_path, shuffle=False, num_classes=10):
        self.data_path = data_path
        self.gt_path = gt_path
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.data_files = [filename for filename in os.listdir(data_path)]
        self.num_samples = len(self.data_files)
        self.blob_list = []
        self.max_gt_count = 0
        self.min_gt_count = sys.maxsize
        self.bin = 0
        self.count_class_hist = np.zeros(num_classes)
        self.load_all()

    def __getitem__(self, item):
        x = self.blob_list[item]['data']
        den = self.blob_list[item]['gt_den']
        cls = self.blob_list[item]['gt_class']
        # 增加batch 维返回
        return x[np.newaxis, ...], {"density": den[np.newaxis, ...], "cls": cls[np.newaxis, ...]}

    def __len__(self):
        return len(self.blob_list)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.blob_list)

    def load_all(self):
        """
        一次性加载所有数据
        :return:
                X, 图片数组, shape(num_samples, h, w, 1);
                Y_den, 密度图GT, shape(num_samples, h, w, 1);
                Y_count, 类别GT, shape(num_samples, num_classes)
        """
        print('[INFO] Loading data, wait a moment...')
        for i, fname in enumerate(self.data_files, 1):
            img = cv2.imread(os.path.join(self.data_path, fname), 0)
            img = img.astype(np.float32, copy=False)
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = ht // 4 * 4
            wd_1 = wd // 4 * 4
            img = cv2.resize(img, (wd_1, ht_1))
            # 加载密度图
            den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'),
                              header=None).values
            den = den.astype(np.float32, copy=False)
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))
            # 人头数
            gt_count = np.sum(den)
            self.min_gt_count = min(self.min_gt_count, gt_count)
            self.max_gt_count = max(self.max_gt_count, gt_count)
            blob = dict()
            blob['data'] = img
            blob['gt_den'] = den
            blob['gt_count'] = gt_count
            blob['fname'] = fname
            self.blob_list.append(blob)

            if i % 100 == 0:
                print('Loaded {}/{} files'.format(i, self.num_samples))
        print('[INFO] Completed loading {} files.'.format(len(self.data_files)))

        self.assign_classes()  # 设置图片类别
        if self.shuffle:
            np.random.shuffle(self.blob_list)
        xs = np.array([blob['data'] for blob in self.blob_list])
        ys_den = np.array([blob['gt_den'] for blob in self.blob_list])
        ys_class = np.array([blob['gt_class'] for blob in self.blob_list])
        return xs, ys_den, ys_class

    def assign_classes(self):
        """
        设置图片gt类别
        """
        self.bin = (self.max_gt_count - self.min_gt_count) / self.num_classes
        for blob in self.blob_list:
            gt_class = np.zeros(self.num_classes, dtype=np.int32)
            idx = np.round(blob['gt_count'] / self.bin)
            idx = int(min(idx, self.num_classes - 1))
            gt_class[idx] = 1
            blob['gt_class'] = gt_class
            self.count_class_hist[idx] += 1

    def get_class_weights(self):
        """
        根据每类的样本数量对每一类设置权重（可在训练期间让模型更多关注样本较少的类别）
        """
        class_weights = 1 - self.count_class_hist / self.num_samples
        class_weights = class_weights / np.sum(class_weights)
        return class_weights

    @staticmethod
    def random_augment(imgs, gts):
        """随机增广"""
        imgs_aug = []
        gts_aug = []
        for img, gt in zip(imgs, gts):
            if np.random.uniform() > 0.5:
                # 随机翻转图片以及density map
                img = np.flip(img, 3).copy()
                gt = np.flip(gt, 3).copy()
            if np.random.uniform() > 0.5:
                # 加入随机噪音
                img = img + np.random.uniform(-10, 10, size=img.shape)
            imgs_aug.append(img)
            gts_aug.append(gt)
        return np.array(imgs_aug), np.array(gts_aug)
