"""
DeepCropMapping - Deep learning-based crop mapping using multi-temporal satellite imagery.

This package provides tools and models for processing satellite imagery data
and training deep learning models for crop type classification.
"""

__version__ = "0.1.0"
__author__ = "DeepCropMapping Team"

# 导入主要模块
from .visual import CropDataVisualizer
from .pre_process import check_data_dimensions, align_data_dimensions

__all__ = ['CropDataVisualizer', 'check_data_dimensions', 'align_data_dimensions']