#!/usr/bin/env python3
"""
dimension_checker.py
数据维度检查和对齐工具模块

提供检查和处理数据集维度不匹配问题的功能
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

# 计算项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_data_dimensions(x_path: Optional[str] = None, 
                         y_path: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    检查x.npy和y.npy的维度匹配情况
    
    Args:
        x_path: x数据文件路径，默认为项目dataset/x.npy
        y_path: y数据文件路径，默认为项目dataset/y.npy 
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含检查结果的字典
            - 'match': bool, 维度是否匹配
            - 'x_shape': tuple, X数据形状
            - 'y_shape': tuple, Y数据形状
            - 'x_spatial': tuple, X空间维度
            - 'y_spatial': tuple, Y空间维度
            - 'differences': tuple, 维度差异
            - 'recommendations': list, 推荐的解决方案
    """
    
    # 设置默认路径
    if x_path is None:
        x_path = PROJECT_ROOT / "dataset" / "x.npy"
    else:
        x_path = Path(x_path)
        
    if y_path is None:
        y_path = PROJECT_ROOT / "dataset" / "y.npy"  
    else:
        y_path = Path(y_path)
    
    if verbose:
        print("Loading data for dimension check...")
    
    try:
        x_data = np.load(x_path)
        y_data = np.load(y_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"数据文件未找到: {e}")
    except Exception as e:
        raise Exception(f"加载数据时出错: {e}")
    
    result = {
        'x_shape': x_data.shape,
        'y_shape': y_data.shape,
        'x_dtype': str(x_data.dtype),
        'y_dtype': str(y_data.dtype)
    }
    
    if verbose:
        print(f"X data shape: {x_data.shape}")
        print(f"Y data shape: {y_data.shape}")
        print(f"X data dtype: {x_data.dtype}")
        print(f"Y data dtype: {y_data.dtype}")
    
    # 检查维度匹配
    if len(x_data.shape) > 2:
        x_spatial = x_data.shape[:2]
        y_spatial = y_data.shape if len(y_data.shape) == 2 else y_data.shape[:2]
        
        result.update({
            'x_spatial': x_spatial,
            'y_spatial': y_spatial
        })
        
        if verbose:
            print(f"\nX spatial dimensions: {x_spatial}")  
            print(f"Y spatial dimensions: {y_spatial}")
        
        if x_spatial != y_spatial:
            # 计算差异
            diff_h = y_spatial[0] - x_spatial[0]
            diff_w = y_spatial[1] - x_spatial[1]
            
            result.update({
                'match': False,
                'differences': (diff_h, diff_w),
                'recommendations': [
                    "crop_y_to_x",    # 将Y裁剪到X的尺寸
                    "pad_x_to_y",     # 将X扩充到Y的尺寸  
                    "crop_to_common"  # 都裁剪到最小公共尺寸
                ]
            })
            
            if verbose:
                print("\n❌ DIMENSION MISMATCH DETECTED!")
                print(f"X spatial: {x_spatial} vs Y spatial: {y_spatial}")
                print(f"Height difference: {diff_h}")
                print(f"Width difference: {diff_w}")
                print("\n推荐的解决方案:")
                print("1. 将Y裁剪到X的尺寸 (推荐)")
                print("2. 将X扩充到Y的尺寸")
                print("3. 都裁剪到最小公共尺寸")
        else:
            result.update({
                'match': True,
                'differences': (0, 0),
                'recommendations': []
            })
            
            if verbose:
                print("\n✅ 空间维度匹配!")
                
    else:
        result.update({
            'match': False,
            'x_spatial': x_data.shape,
            'y_spatial': y_data.shape,
            'differences': None,
            'recommendations': ['check_data_format']
        })
        
        if verbose:
            print("数据结构与预期不符")
    
    return result


def align_data_dimensions(x_data: np.ndarray, 
                         y_data: np.ndarray,
                         method: str = "crop_y_to_x") -> Tuple[np.ndarray, np.ndarray]:
    """
    对齐数据维度
    
    Args:
        x_data: 输入数据X
        y_data: 标签数据Y
        method: 对齐方法
            - "crop_y_to_x": 将Y裁剪到X的尺寸
            - "pad_x_to_y": 将X扩充到Y的尺寸
            - "crop_to_common": 都裁剪到最小公共尺寸
            
    Returns:
        tuple: (对齐后的x_data, 对齐后的y_data)
    """
    
    if len(x_data.shape) < 2 or len(y_data.shape) < 2:
        raise ValueError("数据维度不足，无法进行对齐操作")
    
    x_spatial = x_data.shape[:2]  
    y_spatial = y_data.shape[:2] if len(y_data.shape) > 2 else y_data.shape
    
    if x_spatial == y_spatial:
        warnings.warn("数据维度已经匹配，无需对齐")
        return x_data, y_data
    
    if method == "crop_y_to_x":
        # 将Y裁剪到X的尺寸
        min_h, min_w = min(x_spatial[0], y_spatial[0]), min(x_spatial[1], y_spatial[1])
        x_aligned = x_data[:min_h, :min_w]
        y_aligned = y_data[:min_h, :min_w]
        
    elif method == "pad_x_to_y":
        # 将X扩充到Y的尺寸（用0填充）
        max_h, max_w = max(x_spatial[0], y_spatial[0]), max(x_spatial[1], y_spatial[1])
        
        # 填充X数据
        if len(x_data.shape) == 4:  # (H, W, T, C)
            x_aligned = np.zeros((max_h, max_w, x_data.shape[2], x_data.shape[3]), dtype=x_data.dtype)
            x_aligned[:x_spatial[0], :x_spatial[1]] = x_data
        else:
            x_aligned = np.zeros((max_h, max_w), dtype=x_data.dtype)  
            x_aligned[:x_spatial[0], :x_spatial[1]] = x_data
        
        # 填充Y数据
        y_aligned = np.zeros((max_h, max_w), dtype=y_data.dtype)
        y_aligned[:y_spatial[0], :y_spatial[1]] = y_data
        
    elif method == "crop_to_common":
        # 都裁剪到最小公共尺寸
        min_h = min(x_spatial[0], y_spatial[0])
        min_w = min(x_spatial[1], y_spatial[1]) 
        x_aligned = x_data[:min_h, :min_w]
        y_aligned = y_data[:min_h, :min_w]
        
    else:
        raise ValueError(f"未知的对齐方法: {method}")
    
    return x_aligned, y_aligned


if __name__ == "__main__":
    # 命令行界面
    result = check_data_dimensions()
    
    if not result['match']:
        print(f"\n需要进行数据对齐处理!")
        print(f"可以使用 align_data_dimensions() 函数进行处理")