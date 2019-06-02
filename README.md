# keras-crowdcounting-cmtl

keras复现人群数量估计网络"CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting"。
本工程的实现主要参考[crowdcount-cascaded-mtl](https://github.com/svishwa/crowdcount-cascaded-mtl)和[keras-mcnn](https://github.com/embracesource-cv-com/keras-mcnn)
在ShanghaiTech数据集上训练和测试效果如下：

    |        |  MAE   |  MSE   |
    ----------------------------
    | Part_A |  115.57 |  179.82 |
    ----------------------------
    | Part_B |  26.30  |  48.78  |

## 安装
1. Clone
    ```shell
    git clone https://github.com/embracesource-cv-com/keras-crowdcounting-cmtl
    ```

2. 安装依赖库
    ```shell
    cd keras-mcnn
    pip install -r requirements.txt
    ```

## 数据配置
1. 下载ShanghaiTech数据集:    
    [Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)
    or [百度云盘](http://pan.baidu.com/s/1nuAYslz)

2. 创建数据存放目录$ORIGIN_DATA_PATH
    ```shell
    mkdir /opt/dataset/crowd_counting/shanghaitech/original
    ```

3. 将```part_A_final```和```part_B_final```存放到$ORIGIN_DATA_PATH目录下

4. 生成测试集的ground truth文件
    ```shell
    python create_gt_test_set_shtech.py [A or B]  # Part_A or Part_B
    ```
    生成好的ground-truth文件将会保存在$TEST_GT_PATH/test_data/ground_truth_csv目录下

5. 生成训练集和验证集
    ```shell
    python create_training_set_shtech.py [A or B]
    ```
    生成好的数据保存将会在$TRAIN_PATH、$TRAIN_GT_PATH、$VAL_PATH、$VAL_GT_PATH目录下

## 测试

a)下载与与训练模型

[cmtl-A.235.h5](https://pan.baidu.com/s/1t0vyli5z7f77n2GHdlXqww) 提取码:prxi、[cmtl-B.210.h5](https://pan.baidu.com/s/1n50pxiWAzAJk_LXgZl2heg) 提取码:7if7

b) 如下命令分别测试A和B

```shell
python test.py --dataset A --weight_path /tmp/cmtl-A.235.h5 --output_dir /tmp/ctml_A
python test.py --dataset B --weight_path /tmp/cmtl-B.210.h5 --output_dir /tmp/ctml_B
```


## 训练
如果你想自己训练模型，很简单：
```shell
python train.py [A or B]
```
