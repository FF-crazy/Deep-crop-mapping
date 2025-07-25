# 数据集可视化工具使用说明

## 概述

该工具用于分析和可视化DeepCropMapping项目的卫星遥感数据集，帮助你深入理解数据特征并做出决策。

## 主要功能

### 1. 数据概览 (`data_overview()`)
- 显示数据集基本信息
- 数据类型、维度、内存使用量
- 各作物类别的像素数量和比例统计

### 2. 类别分布分析 (`plot_class_distribution()`)
- 饼图：各作物类别所占比例
- 柱状图：各类别的像素数量
- 直观显示数据集的类别平衡状况

### 3. 空间分布分析 (`plot_spatial_distribution()`)
- 完整标签地图：整个研究区域的作物分布
- 局部放大图：中心区域详细分布
- 行/列统计：不同位置的作物类型分布特征

### 4. 光谱分析 (`plot_spectral_analysis()`)
- 各波段统计特征：均值和方差分布
- 不同作物类型的光谱特征对比
- 光谱数值分布直方图

### 5. 时序分析 (`plot_temporal_analysis()`)
- **NDVI时序曲线**：不同作物的生长季变化模式
- **随机像素时序**：个别像素点的光谱时间变化
- **时序统计**：整体数据的时间变化趋势
- **时序热力图**：多个像素的时间模式可视化
- **季节性分析**：月度NDVI变化
- **时序方差分布**：数据变化的稳定性分析

### 6. 交互式像素探索 (`interactive_pixel_explorer()`)
- 支持指定坐标或随机选择像素点
- 显示该像素点的详细信息：
  - 所有波段的时序曲线
  - NDVI时序变化
  - 光谱-时序热力图  
  - 周围区域的标签分布

### 7. 完整报告生成 (`generate_report()`)
- 自动生成所有可视化图表
- 创建详细的文本分析报告
- 包含数据质量评估和改进建议

## 使用方法

### 基本使用
```python
from deepcropmapping.data_visualization import CropDataVisualizer

# 初始化（数据集路径会自动指向../dataset/）
visualizer = CropDataVisualizer()

# 查看数据概览
visualizer.data_overview()

# 生成各种可视化图表
visualizer.plot_class_distribution()
visualizer.plot_spatial_distribution()
visualizer.plot_spectral_analysis()
visualizer.plot_temporal_analysis()

# 探索特定像素点（例如坐标 (100, 200)）
visualizer.interactive_pixel_explorer(200, 100)

# 生成完整报告
visualizer.generate_report()
```

### 命令行使用
```bash
cd deepcropmapping
python data_visualization.py
```

## 输出文件

运行 `generate_report()` 后会在 `./visualization_output/` 目录下生成：

- `class_distribution.png` - 类别分布图表
- `spatial_distribution.png` - 空间分布图表  
- `spectral_analysis.png` - 光谱分析图表
- `temporal_analysis.png` - 时序分析图表
- `data_analysis_report.txt` - 详细文本报告

## 关键发现

通过可视化分析，你可以获得以下关键信息：

1. **类别不平衡**: 查看各作物类别的数量分布
2. **空间聚集性**: 观察作物分布的空间模式
3. **光谱可分离性**: 评估不同作物的光谱差异
4. **时序变化模式**: 理解各作物的生长周期特征
5. **数据质量**: 识别异常值和噪声

## 辅助决策

基于可视化结果，你可以做出以下决策：

- **数据预处理策略**: 根据异常值分布制定清洗方案
- **类别均衡方法**: 针对不平衡数据选择合适的处理方式
- **特征工程方案**: 基于光谱和时序特征设计新的特征
- **模型架构选择**: 根据数据特点选择合适的深度学习模型
- **训练策略**: 制定针对性的训练和验证策略

## 数据集说明

- **输入数据 (x.npy)**: (326, 1025, 28, 8) - 28期多光谱数据，8个波段
- **标签数据 (y.npy)**: (326, 1025) - 8个作物类别标签
- **时间跨度**: 2021年4月-10月
- **作物类别**: 玉米、小麦、向日葵、番瓜、人造地表、水体、道路、其他

使用这个工具可以帮助你全面了解数据集特征，为后续的模型开发提供重要参考！