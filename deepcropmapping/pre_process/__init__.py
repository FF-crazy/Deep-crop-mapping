"""
数据预处理模块
处理数据集的维度检查、对齐和预处理功能
"""

from .dimension_checker import check_data_dimensions, align_data_dimensions

__all__ = ['check_data_dimensions', 'align_data_dimensions']