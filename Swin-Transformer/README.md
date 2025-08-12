# Swin Transformer for Crop Mapping

基于Swin Transformer的农作物制图模型实现，专门用于多光谱时序卫星数据的农作物分类。Swin Transformer采用层次化设计和移位窗口注意力机制，在保持高精度的同时实现了线性计算复杂度。

## 模型特点

### 架构优势
- **层次化设计**: 类似CNN的多尺度特征提取，适合密集预测任务
- **移位窗口注意力**: 在不重叠窗口内计算自注意力，然后使用移位窗口建立连接
- **线性复杂度**: 计算复杂度相对于输入大小呈线性关系
- **多尺度融合**: 通过Patch Merging实现特征图下采样和通道数增加

### 技术特性
- 支持多光谱时序数据 (8个波段，28个时间步)
- 窗口化注意力机制，计算效率高
- 相对位置编码，保持位置信息
- Drop Path正则化，提升泛化能力
- 支持不同窗口大小的灵活配置

## 文件结构

```
Swin-Transformer/
├── __init__.py              # 模块初始化
├── model.py                 # Swin Transformer模型定义
├── dataset.py               # 数据加载和预处理
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 推理脚本
├── utils.py                 # 工具函数
├── compat.py                # 兼容性工具
└── README.md               # 说明文档
```

## 快速开始

### 1. 训练模型

```bash
# 基本训练
python -m Swin-Transformer.train --data-path ./dataset --epochs 100

# 高级配置训练
python -m Swin-Transformer.train \
    --data-path ./dataset \
    --patch-size 64 \
    --swin-patch-size 4 \
    --embed-dim 96 \
    --depths 2 2 6 2 \
    --num-heads 3 6 12 24 \
    --window-size 7 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --epochs 100 \
    --use-combined-loss \
    --scheduler-type cosine \
    --save-dir ./Swin-Transformer/checkpoints
```

### 2. 评估模型

```bash
python -m Swin-Transformer.evaluate \
    --model-path ./Swin-Transformer/checkpoints/best_model.pth \
    --data-path ./dataset \
    --visualize \
    --analyze-windows
```

### 3. 推理预测

```bash
python -m Swin-Transformer.inference \
    --model-path ./Swin-Transformer/checkpoints/best_model.pth \
    --input-path ./dataset/x.npy \
    --output-dir ./Swin-Transformer/predictions \
    --use-tta \
    --visualize \
    --save-geotiff
```

## 模型参数

### 核心参数
- `embed_dim`: 嵌入维度 (默认: 96)
- `depths`: 每个阶段的层数 (默认: [2, 2, 6, 2])
- `num_heads`: 各阶段注意力头数 (默认: [3, 6, 12, 24])
- `window_size`: 窗口大小 (默认: 7)
- `swin_patch_size`: Swin patch大小 (默认: 4)
- `mlp_ratio`: MLP扩展比例 (默认: 4.0)
- `drop_path_rate`: Drop Path率 (默认: 0.1)

### 训练参数
- `batch_size`: 批次大小 (默认: 8, 可根据GPU内存调整)
- `learning_rate`: 学习率 (默认: 1e-4)
- `weight_decay`: 权重衰减 (默认: 0.05, Swin使用较大值)
- `gradient_accumulation_steps`: 梯度累积步数 (默认: 2)
- `max_grad_norm`: 梯度裁剪阈值 (默认: 5.0)

## 损失函数选择

### 1. 标准交叉熵损失
```bash
python -m Swin-Transformer.train --use-focal-loss False
```

### 2. Focal Loss (处理类别不平衡)
```bash
python -m Swin-Transformer.train --use-focal-loss True --focal-gamma 2.5
```

### 3. 组合损失 (推荐)
```bash
python -m Swin-Transformer.train --use-combined-loss --focal-gamma 2.5 --label-smoothing 0.1
```

## 学习率调度策略

### 1. Cosine Annealing (推荐)
```bash
python -m Swin-Transformer.train --scheduler-type cosine --min-lr 1e-6
```

### 2. OneCycle Learning Rate
```bash
python -m Swin-Transformer.train --scheduler-type onecycle --max-lr 1e-3
```

### 3. Reduce on Plateau
```bash
python -m Swin-Transformer.train --scheduler-type plateau
```

## 数据增强

Swin Transformer版本包含专门的窗口化数据增强策略：

- **窗口级增强**: 在窗口内进行像素混洗
- **移位窗口增强**: 模拟Swin的移位窗口机制
- **窗口Mixup**: 不同窗口间的特征混合
- **时序增强**: 时间步遮掩和光谱增强

## 性能优化

### 1. 窗口大小选择
不同的窗口大小适合不同的数据：
- 小窗口 (window_size=4): 适合小目标，计算快
- 中窗口 (window_size=7): 平衡性能和效率
- 大窗口 (window_size=12): 适合大目标，但计算慢

### 2. 模型规模调整
```bash
# 小模型 (更快训练)
python -m Swin-Transformer.train --embed-dim 64 --depths 2 2 2 2 --num-heads 2 4 8 16

# 中模型 (默认)
python -m Swin-Transformer.train --embed-dim 96 --depths 2 2 6 2 --num-heads 3 6 12 24

# 大模型 (更高精度)
python -m Swin-Transformer.train --embed-dim 128 --depths 2 2 18 2 --num-heads 4 8 16 32
```

### 3. 测试时间增强
```bash
python -m Swin-Transformer.inference --use-tta
```
使用翻转变换的集成预测，提升准确性。

## 监控和可视化

### 1. Weights & Biases集成
```bash
python -m Swin-Transformer.train --use-wandb
```

### 2. 窗口注意力可视化
```bash
python -m Swin-Transformer.evaluate --analyze-windows
```

### 3. 训练曲线和指标
所有训练指标和可视化结果自动保存到输出目录。

## 与其他模型对比

| 特性 | Swin Transformer | TCN | Vision Transformer |
|------|------------------|-----|--------------------|
| 计算复杂度 | O(n) | O(n) | O(n²) |
| 多尺度特征 | 强 | 中等 | 弱 |
| 长距离依赖 | 强 | 中等 | 强 |
| 参数效率 | 高 | 很高 | 中等 |
| 训练稳定性 | 高 | 很高 | 中等 |
| 推理速度 | 快 | 很快 | 中等 |

## 使用建议

### 1. 数据预处理
- 确保输入patch大小是swin_patch_size的倍数
- 对于小目标，使用较小的patch_size和window_size
- 时序数据建议进行标准化

### 2. 训练技巧
- 使用较大的weight_decay (0.05)
- 采用梯度裁剪防止梯度爆炸
- 使用Drop Path提升泛化能力

### 3. 推理优化
- 批量推理时调整batch_size以平衡速度和内存
- 对重要区域可以使用更多重叠
- TTA可以提升精度但会增加计算时间

## 故障排除

### 1. 内存不足
- 减少batch_size或embed_dim
- 使用梯度累积
- 减少window_size

### 2. 训练不稳定
- 降低学习率
- 增加梯度裁剪
- 使用warmup调度器

### 3. 收敛慢
- 检查数据预处理是否正确
- 调整学习率调度策略
- 尝试不同的优化器参数

## 技术细节

### 窗口注意力机制
Swin Transformer的核心创新是移位窗口注意力：
- 第偶数层：使用常规窗口划分
- 第奇数层：窗口移位window_size//2
- 通过相对位置编码保持位置信息

### 层次化特征提取
模型分为4个阶段，每个阶段后分辨率减半，通道数翻倍：
- Stage 1: 分辨率H/4×W/4, 通道C
- Stage 2: 分辨率H/8×W/8, 通道2C  
- Stage 3: 分辨率H/16×W/16, 通道4C
- Stage 4: 分辨率H/32×W/32, 通道8C

这种设计使模型能够捕获不同尺度的特征，特别适合遥感图像中不同大小地物的识别。

## 引用

如果使用本实现，请引用：
```bibtex
@article{swin_transformer_crop_mapping,
  title={Swin Transformer for Multi-Spectral Crop Mapping},
  author={DeepCropMapping Project},
  year={2024}
}
```