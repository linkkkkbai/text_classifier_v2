# Text Region Attribute Classifier

一个基于深度学习的文本区域属性分类器，用于识别图像中文本区域的多个属性，包括类型（可编辑/不可编辑）、字体（常规/粗体）和斜体（是/否）。

## 功能特点

- 基于ResNet50的多属性分类器
- 支持三种文本属性的同时分类：
  - 类型（Type）：可编辑/不可编辑
  - 字体（Font）：常规/粗体
  - 斜体（Italic）：是/否
- 使用COCO格式的数据集
- 支持数据增强和标准化处理
- 提供完整的训练、评估和预测功能

## 安装要求

```bash
# 主要依赖
torch>=1.7.0
torchvision>=0.8.0
opencv-python>=4.5.0
numpy>=1.19.0
timm>=0.4.12
PyYAML>=5.4.0
```

## 数据格式

项目使用COCO格式的数据集，需要包含以下内容：

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "path/to/image.jpg",
            "height": 1080,
            "width": 1920
        }
    ],
    "annotations": [
        {
            "image_id": 1,
            "bbox": [x, y, width, height],
            "attributes": {
                "类型": "可编辑/不可编辑",
                "字体": "常规/粗体",
                "斜体": "是/否"
            }
        }
    ]
}
```

## 模型架构

模型基于ResNet50骨干网络，添加了三个独立的分类头：

```
ResNet50 Backbone
    │
    ├── Type Classification Head
    │       └── FC(2048 -> num_type_classes)
    │
    ├── Font Classification Head
    │       └── FC(2048 -> num_font_classes)
    │
    └── Italic Classification Head
            └── FC(2048 -> num_italic_classes)
```

## 配置文件

在 `configs/base.yaml` 中设置模型参数和训练配置：

```yaml
model:
  pretrained: true  # 是否使用预训练权重

data:
  root_dir: "path/to/dataset"  # 数据集根目录
  train_json: "train.json"     # 训练集标注文件
  val_json: "val.json"         # 验证集标注文件

attributes:
  type:
    classes: ["可编辑", "不可编辑"]
  font:
    classes: ["常规", "粗体"]
  italic:
    classes: ["否", "是"]

train:
  batch_size: 32
  num_workers: 4
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
```

## 使用说明

### 训练模型

```bash
python train.py --config configs/base.yaml
```

### 评估模型

```bash
python evaluate.py --config configs/base.yaml --checkpoint path/to/checkpoint.pth
```

### 可视化预测结果

```bash
python visual_predictions.py --config configs/base.yaml --checkpoint path/to/checkpoint.pth --image path/to/image.jpg
```

## 数据预处理

训练时的数据增强包括：
- 随机水平和垂直翻转
- 颜色抖动（亮度、对比度、饱和度、色调）
- 图像大小调整
- 标准化

测试时只进行图像大小调整和标准化。

## 目录结构

```
text_classifier_v2/
├── configs/
│   └── base.yaml          # 配置文件
├── datasets/
│   └── coco_loader.py     # 数据加载器
├── losses/
│   └── focal_loss.py      # 损失函数
├── models/
│   └── multi_head.py      # 模型定义
├── utils/
│   └── train_utils.py     # 工具函数
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
└── visual_predictions.py  # 预测可视化
```

## 注意事项

1. 确保数据集按COCO格式组织
2. 根据实际情况修改配置文件中的路径
3. 建议使用GPU进行训练
4. 可以根据需要调整数据增强策略
