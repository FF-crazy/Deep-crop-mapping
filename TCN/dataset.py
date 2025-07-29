#!/usr/bin/env python3
"""
数据加载和预处理模块
处理多光谱时序数据的加载、预处理和批处理
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any
import pickle
from pathlib import Path


class CropMappingDataset(Dataset):
    """
    农作物制图数据集类
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        patch_size: int = 64,
        stride: int = 32,
        transform: Optional[Any] = None,
        normalize: bool = True,
        scaler: Optional[StandardScaler] = None
    ):
        """
        初始化数据集
        
        Args:
            x_data: 输入数据 (height, width, temporal, spectral)
            y_data: 标签数据 (height, width)
            patch_size: 切片大小
            stride: 切片步长
            transform: 数据增强
            normalize: 是否标准化
            scaler: 预训练的标准化器
        """
        self.x_data = x_data
        self.y_data = y_data
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        
        # 数据标准化
        if self.normalize:
            height, width, temporal, spectral = self.x_data.shape
            x_reshaped = self.x_data.reshape(-1, spectral)
            
            if scaler is None:
                self.scaler = StandardScaler()
                x_normalized = self.scaler.fit_transform(x_reshaped)
            else:
                self.scaler = scaler
                x_normalized = self.scaler.transform(x_reshaped)
            
            self.x_data = x_normalized.reshape(height, width, temporal, spectral)
        
        # 计算可用的切片位置
        self.patches = self._calculate_patches()
        
    def _calculate_patches(self) -> list:
        """计算所有可用的切片位置"""
        patches = []
        h, w = self.y_data.shape
        
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                # 检查切片是否包含有效数据（有效标签1-8）
                patch_labels = self.y_data[i:i+self.patch_size, j:j+self.patch_size]
                if np.any((patch_labels >= 1) & (patch_labels <= 8)):  # 包含有效标签像素
                    patches.append((i, j))
        
        return patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j = self.patches[idx]
        
        # 提取切片
        x_patch = self.x_data[i:i+self.patch_size, j:j+self.patch_size]
        y_patch = self.y_data[i:i+self.patch_size, j:j+self.patch_size]
        
        # 将标签从1-8转换为0-7（PyTorch交叉熵损失要求从0开始）
        y_patch = y_patch - 1
        
        # 转换为张量
        x_patch = torch.from_numpy(x_patch).float()
        y_patch = torch.from_numpy(y_patch).long()
        
        # 数据增强
        if self.transform is not None:
            x_patch = self.transform(x_patch)
        
        return x_patch, y_patch


class DataAugmentation:
    """数据增强类"""
    
    @staticmethod
    def random_flip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """随机翻转"""
        if torch.rand(1) < p:
            # 水平翻转
            x = torch.flip(x, dims=[1])
        if torch.rand(1) < p:
            # 垂直翻转
            x = torch.flip(x, dims=[0])
        return x
    
    @staticmethod
    def random_rotation(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """随机旋转90度的倍数"""
        if torch.rand(1) < p:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[0, 1])
        return x
    
    @staticmethod
    def add_gaussian_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(x) * std
        return x + noise


def prepare_data(
    data_path: str = "./dataset",
    patch_size: int = 64,
    stride: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    准备数据加载器
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        info: 数据集信息字典
    """
    # 加载数据
    x_data = np.load(f"{data_path}/x.npy")
    y_data = np.load(f"{data_path}/y.npy")
    
    print(f"Original data shape - X: {x_data.shape}, Y: {y_data.shape}")
    
    # 计算类别权重（处理类别不平衡）
    # 只考虑有效标签1-8，转换为0-7后计算权重
    valid_mask = (y_data >= 1) & (y_data <= 8)
    valid_labels = y_data[valid_mask] - 1  # 转换为0-7
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    class_weights = len(valid_labels) / (len(unique_labels) * counts)
    class_weights = torch.FloatTensor(class_weights)
    
    # 创建数据集
    full_dataset = CropMappingDataset(
        x_data, y_data, 
        patch_size=patch_size,
        stride=stride,
        normalize=True
    )
    
    # 分割数据集
    total_samples = len(full_dataset)
    indices = list(range(total_samples))
    
    # 首先分出测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # 再从训练验证集中分出验证集
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 保存数据信息
    info = {
        'num_classes': len(unique_labels),
        'class_weights': class_weights,
        'class_names': {
            1: 'Corn',
            2: 'Wheat', 
            3: 'Sunflower',
            4: 'Pumpkin',
            5: 'Artificial_Surface',
            6: 'Water',
            7: 'Road',
            8: 'Other'
        },
        'scaler': full_dataset.scaler if hasattr(full_dataset, 'scaler') else None,
        'patch_size': patch_size,
        'input_shape': (patch_size, patch_size, x_data.shape[2], x_data.shape[3])
    }
    
    return train_loader, val_loader, test_loader, info


def save_data_info(info: Dict[str, Any], save_path: str = "./TCN/data_info.pkl"):
    """保存数据信息"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(info, f)
    print(f"Data info saved to {save_path}")


def load_data_info(load_path: str = "./TCN/data_info.pkl") -> Dict[str, Any]:
    """加载数据信息"""
    with open(load_path, 'rb') as f:
        info = pickle.load(f)
    return info


if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader, info = prepare_data(
        batch_size=4,
        patch_size=32
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class weights: {info['class_weights']}")
    
    # 测试一个批次
    for batch_x, batch_y in train_loader:
        print(f"\nBatch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")
        print(f"X range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"Y unique: {torch.unique(batch_y)}")
        break