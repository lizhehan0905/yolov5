# 训练相关基本操作
## 环境
+ python
+ torch
+ 其他
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```
## 数据集
+ 数据格式
```
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

```
## 命令行执行
### 训练必用train.py
+ --weights 默认yolov5s迁移学习，自动下载，输入false则默认加载--cfg中的配置文件
+ --cfg 默认在model文件夹下找yaml文件，修改yaml文件实现多种模型子模块的增删改查
+ --data 数据集加载，默认data文件夹下找yaml文件
+ --hyp 超参，默认data/hyps超参下找yaml
+ --epochs 训练代数
+ --batch-size 视显存大小而定
+ --img 训练尺度
+ --resume 断点重训
+ --device gpu数量
+ --workers 0肯定可以，其他数值请自行尝试
+ --patience epochs设置比较大的时候可以用到early-stop

### 验证必用val.py
+ --data 数据集加载，默认data文件夹下找yaml文件
+ --weights 想要验证的权重文件地址，支持pt、torchscript、onnx、openvino、trt、tf、paddle
+ --batch-size 视验证目标而定
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --task 主要用val、test、speed
+ --device gpu数量
+ --workers 0肯定可以，其他数值请自行尝试
+ --half fp16推理

### 推理必用detect.py
+ --weights 想要验证的权重文件地址，支持pt、torchscript、onnx、openvino、trt、tf、paddle
+ --source 想要推理的目录，可以是图片、视频、文件夹、屏幕、摄像头
+ --img 推理尺度
+ --conf-thres 置信度阈值，影响较大
+ --iou-thres iou阈值
+ --max-det 最大检测数量
+ --device gpu数量
+ --nosave 不保存

### 导出必用export.py
+ --weights 想要导出的权重地址
+ --img 导出尺度
+ --batch-size 导出尺度
+ --device gpu数量
+ --half 半精度导出
+ --dynamic 动态导出
+ --simplify 调用onnx-simplify进行简化
+ --opset onnx版本
+ --include 导出格式，主要有onnx，torchscript，engine等等

### 全面验证benchmarks.py
+ torch、torchscript、onnx、openvino、trt、coreml、tf等等全部跑一遍
+ --weights 想要导出的权重地址
+ --img 导出尺度
+ --batch-size 导出尺度
+ --device gpu数量
+ --half 半精度导出
+ --pt-only 仅测试pt

# 代码基础介绍
## data数据代码
### hyps超参
+ 学习率
+ 动量参数
+ warmup参数
+ 分类损失与回归损失的权重参数
+ 数据增强参数
### 数据集.yaml
+ 训练集/验证集/测试集地址
+ 名称
## models模块
+ common.py子模块实现代码，例如卷积、池化等等
+ yolo.py子类实现代码，例如检测、分割等基础模型
+ yaml文件，配置类别（自动检测）、宽度、深度、anchor、子模块调用
## utils工具
+ activation.py自定义的激活函数
+ augmentations.py自定义的数据增强
+ autoanchor.py自动计算anchor
+ autobatch.py自动计算batch
+ dataloaders.py数据加载
+ downloads.py下载工具
+ general.py常规工具
+ loss.py损失函数
+ metrics.py AP的计算工具
+ plot.py绘图工具
+ torch_utils.py
+ triton.py
## classify分类的调用代码
## segment分割的调用代码

# 改进方法
## 训练方式
+ 迁移学习：数据集<10w张，建议从官方权重开始迁移学习
```
python train.py --weigths yolov5s.pt
```
+ 从零训练:数据量大，建议从零训练，epoch可以适当多设置一些
```
python train.py --weights false --cfg model/yolov5s.yaml
```
## 训练尺度
+ 默认640，可以按照实际需求设置，例如512、480、320，也可以按照论文推荐的640-1280-640的方式来训练，提高训练效果
```
python train.py --img 320
```
## 激活函数的替换
+ 在model/common.py中替换掉卷积的激活函数，可以用relu、Lrelu、hardswish等等
+ 如果不是torch内置的激活函数，可以将定义添加到common.py中，或者添加到utils/activations.py中调用
## 宽度高度的调整
+ 在models文件夹中添加自己的yaml文件，更改类别数，更改宽度系数和高度系数，可以快速调整整个模型的卷积堆叠个数和filter个数
## 模型架构调整
+ 例如p2模型和p6模型，分别用于小目标检测与超大图检测
+ 同时可以在模型中插入、删除、改进部分模块，例如conv替换为dwconv，例如增加注意力机制，例如backbone整个替换
## 超参调整
+ 例如学习率、损失权重、数据增强