# TCN农作物制图模型

基于时序卷积网络(Temporal Convolutional Network)的农作物制图深度学习模型。

## 项目结构

```
TCN/
├── model.py                          # TCN模型架构定义
├── dataset.py                        # 数据加载和预处理
├── train.py                          # 命令行训练脚本
├── evaluate.py                       # 模型评估脚本
├── inference.py                      # 推理脚本
├── utils.py                          # 工具函数
├── TCN_training_and_evaluation.ipynb # 完整训练和评估notebook
├── README.md                         # 说明文档
├── checkpoints/                      # 模型检查点目录
└── evaluation/                       # 评估结果目录
```

## 快速开始

### 1. 环境准备

确保已安装必要的依赖包：
```python
torch>=1.8.0
numpy
matplotlib
scikit-learn
seaborn
pandas
tqdm
rasterio
```

### 2. 使用Jupyter Notebook训练（推荐）

打开并运行 `TCN_training_and_evaluation.ipynb`:

```bash
jupyter notebook TCN_training_and_evaluation.ipynb
```

该notebook包含完整的训练、评估和推理流程，适合交互式使用。

### 3. 命令行训练

```bash
python train.py \
    --data-path ../dataset \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --patch-size 64 \
    --save-dir ./checkpoints
```

### 4. 模型评估

```bash
python evaluate.py \
    --model-path ./checkpoints/best_model.pth \
    --data-path ../dataset \
    --save-dir ./evaluation \
    --visualize
```

### 5. 推理预测

```bash
python inference.py \
    --model-path ./checkpoints/best_model.pth \
    --input-path ../dataset/x.npy \
    --output-dir ./predictions \
    --visualize \
    --save-geotiff
```

## 模型特点

- **时序建模**: 使用TCN处理28个时间步的多光谱数据
- **因果卷积**: 保证时序数据的因果性
- **残差连接**: 提高模型训练稳定性
- **类别平衡**: 使用Focal Loss处理类别不平衡问题
- **混合精度**: 支持加速训练和减少显存占用

## 数据格式

- **输入数据** (x.npy): 形状为 `(height, width, time_steps, spectral_bands)`
- **标签数据** (y.npy): 形状为 `(height, width)`
- **类别映射**: 9个农作物类别 (0-8)

## 性能指标

模型使用以下指标进行评估：
- **总体准确率** (Overall Accuracy)
- **平均交并比** (Mean IoU)
- **类别准确率** (Per-class Accuracy)
- **混淆矩阵** (Confusion Matrix)

## 配置参数

主要配置参数可在notebook或训练脚本中调整：

```python
config = {
    'patch_size': 64,        # 切片大小
    'batch_size': 16,        # 批次大小
    'learning_rate': 1e-3,   # 学习率
    'tcn_channels': [64, 128, 256],  # TCN通道数
    'epochs': 100,           # 训练轮数
    'dropout': 0.2,          # Dropout率
}
```

## 文件输出

训练完成后会生成以下文件：
- `best_model.pth`: 最佳模型权重
- `history.json`: 训练历史记录
- `data_info.pkl`: 数据预处理信息
- `training_curves.png`: 训练曲线图
- `confusion_matrix.png`: 混淆矩阵图
- `training_summary.txt`: 训练总结报告

## 注意事项

1. 确保GPU内存足够，推荐8GB以上
2. 数据预处理包含标准化，确保数据分布合理
3. 模型使用滑动窗口进行推理，可能需要较长时间
4. 建议使用Jupyter Notebook进行交互式训练和调试