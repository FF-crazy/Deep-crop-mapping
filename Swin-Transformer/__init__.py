"""
Swin Transformer农作物制图模型包
Swin Transformer for Crop Mapping

主要模块：
- model: Swin Transformer模型架构
- dataset: 数据加载和预处理
- train: 训练相关功能
- evaluate: 评估功能
- inference: 推理功能
- utils: 工具函数
"""

from .model import CropMappingSwinTransformer, create_swin_model
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
    'CropMappingSwinTransformer',
    'create_swin_model',
    'CropMappingDataset',
    'prepare_data',
    'save_checkpoint',
    'load_checkpoint',
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_training_history'
]