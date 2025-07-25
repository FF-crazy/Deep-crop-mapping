"""
TCN农作物制图模型包
Temporal Convolutional Network for Crop Mapping

主要模块：
- model: TCN模型架构
- dataset: 数据加载和预处理
- train: 训练相关功能
- evaluate: 评估功能
- inference: 推理功能
- utils: 工具函数
"""

from .model import CropMappingTCN, create_tcn_model
from .dataset import CropMappingDataset, prepare_data
from .utils import (
    save_checkpoint,
    load_checkpoint, 
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_history
)

__version__ = "1.0.0"
__author__ = "DeepCropMapping Project"

__all__ = [
    'CropMappingTCN',
    'create_tcn_model',
    'CropMappingDataset',
    'prepare_data',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_training_history'
]