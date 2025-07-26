#!/usr/bin/env python3
"""
兼容性工具模块
提供跨版本兼容的工具函数
"""

import torch
from typing import Any, Dict


def load_checkpoint(filepath: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """
    兼容性checkpoint加载函数
    自动处理不同PyTorch版本的weights_only参数
    """
    try:
        # 尝试使用weights_only=False（PyTorch 2.6+）
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        # 如果不支持weights_only参数，使用旧版本语法（PyTorch < 2.6）
        checkpoint = torch.load(filepath, map_location=map_location)
    except Exception as e:
        # 如果仍然失败，可能是pickle问题，尝试添加pickle模块
        import pickle
        checkpoint = torch.load(filepath, map_location=map_location, pickle_module=pickle)
    
    return checkpoint


def get_torch_version():
    """获取PyTorch版本信息"""
    return torch.__version__


def is_torch_version_compatible(min_version: str = "2.0.0") -> bool:
    """
    检查PyTorch版本兼容性
    
    Args:
        min_version: 最低要求版本
        
    Returns:
        是否兼容
    """
    from packaging import version
    current = version.parse(torch.__version__)
    required = version.parse(min_version)
    return current >= required