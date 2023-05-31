# frcnn_fcos
### 准备环境
```bash
pip install -r requirements.txt
```

### 准备数据集
链接：https://pan.baidu.com/s/15KBnpV8CKt_XBl2rZ_X1DA 提取码：eq34，存放成如下的目录。
```
├──frcnn_fcos
  ├── VOCdevkit
    ├── VOC2012
      ├──Annotations
      ├──ImageSets
      ├──JPEGImages
      ├──SegmentationClass
      ├──SegmentationObject
```

### 下载预训练权重
链接：https://pan.baidu.com/s/1VOuZMmZL0cl9WH_99QiU_A 提取码：rj72 将pretrain.pth，resnet50.pth存放在save_weights目录下

### 训练faster rcnn
```bash
python train_mobilenetv2.py
```
训练权重保存在save_weights/frcnn目录下

### 查看frcnn训练结果
```bash
cd runs
tensorboard --logdir frcnn
```

### 测试frcnn
下载训练好的权重，修改名称为frcnn-model.pth，并存放在save_weights/frcnn目录下
```bash
python predict_frcnn.py
```
测试结果保存在同目录下，文件名为test_result.jpg


### 训练fcos
```bash
python train_fcos.py
```
训练权重保存在save_weights/fcos目录下

### 查看fcos训练结果
```bash
cd runs
tensorboard --logdir fcos
```

### 测试fcos
下载训练好的权重，修改名称为fcos-model.pth，并存放在save_weights/fcos目录下
```bash
train predict_fcos.py
```
测试结果保存在同目录下，文件名为test_result.jpg
