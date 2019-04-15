# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import os
import sys


class DataLoader(object):
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, num_classes=10):
        self.data_path = data_path
        self.gt_path = gt_path
        self.shuffle = shuffle
        self.gt_downsample = gt_downsample
        self.num_classes = num_classes
        self.data_files = [filename for filename in os.listdir(data_path)]
        self.num_samples = len(self.data_files)
        self.blob_list = []
        self.max_gt_count = 0
        self.min_gt_count = sys.maxsize
        self.bin = 0
        self.count_class_hist = np.zeros(num_classes)

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
            img = img.reshape((img.shape[0], img.shape[1], 1))
            den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'),
                              header=None).values
            den = den.astype(np.float32, copy=False)
            if self.gt_downsample:
                wd_1 = wd_1 // 4
                ht_1 = ht_1 // 4
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))
            den = den.reshape((den.shape[0], den.shape[1], 1))
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
        print('[INFO] Completed loading {} files.'.format(i))

        self.assign_classes()  # 设置图片类别
        if self.shuffle:
            np.random.shuffle(self.blob_list)
        X = np.array([blob['data'] for blob in self.blob_list])
        Y_den = np.array([blob['gt_den'] for blob in self.blob_list])
        Y_class = np.array([blob['gt_class'] for blob in self.blob_list])
        return X, Y_den, Y_class

    def assign_classes(self):
        """
        设置图片gt类别
        """
        self.bin = (self.max_gt_count - self.min_gt_count) / self.num_classes
        for blob in self.blob_list:
            gt_class = np.zeros(self.num_classes, dtype=np.int32)
            idx = np.round(blob['gt_count'] / self.bin)
            idx = int(min(idx, self.num_classes-1))
            gt_class[idx] = 1
            blob['gt_class'] = gt_class
            self.count_class_hist[idx] += 1

    def get_class_weights(self):
        """
        根据每类的样本数量对每一类设置权重（可在训练期间让模型更多关注样本较少的类别）
        """
        class_weights = 1 - self.count_class_hist / self.num_samples
        class_weights = class_weights / self.num_samples
        return class_weights

    def flow(self, batch_size=32):
        # todo: generate data in batch to save memory
        loop_count = self.num_samples // batch_size
        while True:
            np.random.shuffle(self.blob_list)
            for i in range(loop_count):
                blobs = self.blob_list[i*batch_size: (i+1)*batch_size]
                X = np.array([blob['data'] for blob in blobs])
                Y_den = np.array([blob['gt_den'] for blob in blobs])
                X, Y_den = self.random_augment(X, Y_den)  # 随机增广
                Y_class = np.array([blob['gt_class'] for blob in blobs])
                yield X, Y_den, Y_class

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

    def __iter__(self):
        for blob in self.blob_list:
            yield blob
