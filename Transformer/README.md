# Vision Transformer for Crop Mapping

基于Vision Transformer的农作物制图模型实现，专门用于多光谱时序卫星数据的农作物分类。

## 模型特点

### 架构优势
- **时空融合**: 创新的patch embedding设计，同时处理空间和时间维度
- **自注意力机制**: 多头自注意力捕获长距离依赖关系
- **位置编码**: 保持空间位置信息
- **端到端训练**: 从原始多光谱数据直接输出分类结果

### 技术特性
- 支持多光谱时序数据 (8个波段，28个时间步)
- 自适应patch size用于不同分辨率输入
- 混合精度训练加速
- 梯度累积支持大批次训练
- 测试时间增强(TTA)提升性能

## 文件结构

```
Transformer/
├── __init__.py              # 模块初始化
├── model.py                 # Transformer模型定义
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
python -m Transformer.train --data-path ./dataset --epochs 100

# 高级配置训练
python -m Transformer.train \
    --data-path ./dataset \
    --patch-size 64 \
    --model-patch-size 8 \
    --embed-dim 256 \
    --num-layers 6 \
    --num-heads 8 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --epochs 100 \
    --use-combined-loss \
    --scheduler-type cosine \
    --save-dir ./Transformer/checkpoints
```

### 2. 评估模型

```bash
python -m Transformer.evaluate \
    --model-path ./Transformer/checkpoints/best_model.pth \
    --data-path ./dataset \
    --visualize \
    --analyze-attention
```

### 3. 推理预测

```bash
python -m Transformer.inference \
    --model-path ./Transformer/checkpoints/best_model.pth \
    --input-path ./dataset/x.npy \
    --output-dir ./Transformer/predictions \
    --use-tta \
    --visualize \
    --save-geotiff
```

## 模型参数

### 核心参数
- `embed_dim`: 嵌入维度 (默认: 256)
- `num_layers`: Transformer层数 (默认: 6)
- `num_heads`: 注意力头数 (默认: 8)
- `model_patch_size`: 模型patch大小 (默认: 8)
- `mlp_ratio`: MLP扩展比例 (默认: 4.0)
- `dropout`: Dropout率 (默认: 0.1)

### 训练参数
- `batch_size`: 批次大小 (默认: 8, 可根据GPU内存调整)
- `learning_rate`: 学习率 (默认: 1e-4)
- `weight_decay`: 权重衰减 (默认: 1e-4)
- `gradient_accumulation_steps`: 梯度累积步数 (默认: 2)
- `max_grad_norm`: 梯度裁剪阈值 (默认: 1.0)

## 损失函数选择

### 1. 标准交叉熵损失
```bash
python -m Transformer.train --use-focal-loss False
```

### 2. Focal Loss (处理类别不平衡)
```bash
python -m Transformer.train --use-focal-loss True --focal-gamma 2.5
```

### 3. 组合损失 (推荐)
```bash
python -m Transformer.train --use-combined-loss --focal-gamma 2.5 --label-smoothing 0.1
```

## 学习率调度策略

### 1. Cosine Annealing (推荐)
```bash
python -m Transformer.train --scheduler-type cosine --min-lr 1e-6
```

### 2. OneCycle Learning Rate
```bash
python -m Transformer.train --scheduler-type onecycle --max-lr 1e-3
```

### 3. Reduce on Plateau
```bash
python -m Transformer.train --scheduler-type plateau
```

## 数据增强

Transformer版本包含专门的数据增强策略：

- **时序遮掩**: 随机遮掩某些时间步 (类似BERT masking)
- **光谱dropout**: 随机dropout某些光谱通道
- **空间变换**: 旋转、翻转、patch混洗
- **噪声注入**: 高斯噪声增强鲁棒性

## 性能优化

### 1. 混合精度训练
自动启用，减少GPU内存使用，加速训练。

### 2. 梯度累积
```bash
python -m Transformer.train --gradient-accumulation-steps 4 --batch-size 4
```
等效于batch size 16，但内存需求更少。

### 3. 测试时间增强
```bash
python -m Transformer.inference --use-tta
```
使用多种变换的集成预测，提升准确性。

## 监控和可视化

### 1. Weights & Biases集成
```bash
python -m Transformer.train --use-wandb
```

### 2. 注意力可视化
```bash
python -m Transformer.evaluate --analyze-attention
```

### 3. 训练曲线和指标
所有训练指标和可视化结果自动保存到输出目录。

## 与TCN模型对比

| 特性 | Transformer | TCN |
|------|-------------|-----|
| 时序建模 | 全局自注意力 | 因果卷积 |
| 计算复杂度 | O(n²) | O(n) |
| 长距离依赖 | 强 | 中等 |
| 参数量 | 较多 | 较少 |
| 收敛速度 | 较慢 | 较快 |
| 最终精度 | 通常更高 | 较高 |

## 注意事项

1. **内存需求**: Transformer比TCN需要更多GPU内存，建议使用16GB以上显卡
2. **训练时间**: 收敛较慢，建议使用更多epochs
3. **超参调优**: 对学习率和dropout更敏感，需要仔细调参
4. **数据量**: 在大数据集上表现更好，小数据集可能过拟合

## 故障排除

### 1. 内存不足
- 减少batch size
- 增加gradient accumulation steps  
- 减少embed_dim或num_layers

### 2. 训练不稳定
- 降低学习率
- 增加gradient clipping
- 使用warmup学习率调度

### 3. 过拟合
- 增加dropout率
- 使用更强的数据增强
- 添加权重衰减

## 扩展和定制

模型设计为模块化，可以轻松扩展：
- 修改patch embedding策略
- 添加新的注意力机制
- 集成其他预训练模型
- 适配不同的输入格式

详细的API文档请参考代码注释。