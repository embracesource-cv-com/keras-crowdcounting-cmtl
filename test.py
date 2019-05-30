# -*- coding: utf-8 -*-
"""
   File Name:     test.py
   Description:   测试
   Author:        steven.yi
   date:          2019/04/17
"""
from model import CMTL
from utils.data_loader import DataLoader
import numpy as np
from config import current_config as cfg
import argparse
import os


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = args.dataset  # 'A' or 'B'
    output_dir = args.output_dir
    weight_path = args.weight_path
    cfg.init_path(dataset)

    heatmaps_dir = os.path.join(output_dir, 'heatmaps')  # directory to save heatmap
    results_txt = os.path.join(output_dir, 'results.txt')  # file to save predicted results
    for _dir in [output_dir, heatmaps_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    # load test set
    data_loader = DataLoader(cfg.TEST_PATH,
                             cfg.TEST_GT_PATH,
                             shuffle=False)
    # load model
    print('[INFO] Load model ...')
    model = CMTL(input_shape=(None, None, 1))
    model.load_weights(weight_path, by_name=True)

    # test
    print('[INFO] Testing Part_{} ...'.format(dataset))
    mae = 0.0
    mse = 0.0
    acc = 0.0
    for blob in data_loader.blob_list:
        img = blob['data']
        gt_den = blob['gt_den']
        gt_cls = np.argmax(blob['gt_class'])
        pred_den, pred_cls = model.predict(img[np.newaxis, ...])
        if np.argmax(pred_cls[0]) == gt_cls:
            acc += 1
        gt_count = np.sum(gt_den)
        pred_count = np.sum(pred_den)
        mae += abs(gt_count - pred_count)
        mse += ((gt_count - pred_count) * (gt_count - pred_count))
        # # create and save heatmap
        # pred = np.squeeze(pred)  # shape(1, h, w, 1) -> shape(h, w)
        # save_heatmap(pred, blob, test_path, heatmaps_dir)
        # save results
        with open(results_txt, 'a') as f:
            line = '<{}> {:.2f}--{:.2f}\t{}--{}\n'.format(blob['fname'].split('.')[0], gt_count, pred_count,
                                                          gt_cls, np.argmax(pred_cls[0]))
            f.write(line)

    mae = mae / data_loader.num_samples
    mse = np.sqrt(mse / data_loader.num_samples)
    acc = acc / data_loader.num_samples
    print('[RESULT] MAE: %0.2f, MSE: %0.2f, Acc: %0.2f' % (mae, mse, acc))
    with open(results_txt, 'a') as f:
        f.write('MAE: %0.2f, MSE: %0.2f, Acc: %0.2f' % (mae, mse, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="A", help="the dataset you want to predict", choices=['A', 'B'])
    parser.add_argument("--weight_path", type=str, default="tmp/cmtl-A.h5", help="weight path")
    parser.add_argument("--output_dir", default="./", help="output directory to save predict heatmaps")

    args = parser.parse_args()
    main(args)
