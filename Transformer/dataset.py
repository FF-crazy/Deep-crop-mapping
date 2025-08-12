#!/usr/bin/env python3
"""
数据加载和预处理模块 - Transformer版本
处理多光谱时序数据的加载、预处理和批处理，适配Transformer模型
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
    农作物制图数据集类 - Transformer版本
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        patch_size: int = 64,
        stride: int = 32,
        transform: Optional[Any] = None,
        normalize: bool = True,
        scaler: Optional[StandardScaler] = None,
        augment: bool = False
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
            augment: 是否使用数据增强
        """
        self.x_data = x_data
        self.y_data = y_data
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        
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
                if np.any((patch_labels >= 1) & (patch_labels <= 8)):
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
        
        # 数据增强 - Transformer版本
        if self.augment:
            x_patch, y_patch = self._apply_augmentation(x_patch, y_patch)
        
        # 其他变换
        if self.transform is not None:
            x_patch = self.transform(x_patch)
        
        return x_patch, y_patch
    
    def _apply_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用数据增强（适配Transformer）
        """
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        
        # 随机垂直翻转
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[0])
            y = torch.flip(y, dims=[0])
        
        # 随机旋转90度的倍数
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[0, 1])
            y = torch.rot90(y, k, dims=[0, 1])
        
        # 时序增强：随机dropout一些时间步
        if torch.rand(1) < 0.3:
            temporal_steps = x.shape[2]
            mask_ratio = 0.1  # 10%的时间步
            num_mask = int(temporal_steps * mask_ratio)
            mask_indices = torch.randperm(temporal_steps)[:num_mask]
            x[:, :, mask_indices, :] = 0
        
        # 光谱增强：添加轻微的高斯噪声
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y


class TransformerDataAugmentation:
    """Transformer专用数据增强类"""
    
    @staticmethod
    def temporal_masking(x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        """
        时序遮掩增强（类似BERT的masking）
        """
        batch_size, height, width, temporal, spectral = x.shape
        num_mask = int(temporal * mask_ratio)
        
        for b in range(batch_size):
            mask_indices = torch.randperm(temporal)[:num_mask]
            x[b, :, :, mask_indices, :] = 0
        
        return x
    
    @staticmethod
    def spectral_dropout(x: torch.Tensor, dropout_ratio: float = 0.1) -> torch.Tensor:
        """
        光谱通道dropout
        """
        batch_size, height, width, temporal, spectral = x.shape
        num_drop = int(spectral * dropout_ratio)
        
        for b in range(batch_size):
            drop_indices = torch.randperm(spectral)[:num_drop]
            x[b, :, :, :, drop_indices] = 0
        
        return x
    
    @staticmethod
    def patch_shuffle(x: torch.Tensor, shuffle_ratio: float = 0.1) -> torch.Tensor:
        """
        空间patch混洗
        """
        batch_size, height, width, temporal, spectral = x.shape
        patch_size = 4  # 小patch大小
        
        if height % patch_size != 0 or width % patch_size != 0:
            return x  # 如果不能整除，跳过
        
        patch_h = height // patch_size
        patch_w = width // patch_size
        num_patches = patch_h * patch_w
        num_shuffle = int(num_patches * shuffle_ratio)
        
        for b in range(batch_size):
            # 重塑为patches
            x_patches = x[b].view(patch_h, patch_size, patch_w, patch_size, temporal, spectral)
            x_patches = x_patches.permute(0, 2, 1, 3, 4, 5).contiguous()
            x_patches = x_patches.view(num_patches, patch_size, patch_size, temporal, spectral)
            
            # 随机混洗一些patches
            if num_shuffle > 0:
                shuffle_indices = torch.randperm(num_patches)[:num_shuffle]
                shuffled_patches = x_patches[shuffle_indices][torch.randperm(num_shuffle)]
                x_patches[shuffle_indices] = shuffled_patches
            
            # 重塑回原始形状
            x_patches = x_patches.view(patch_h, patch_w, patch_size, patch_size, temporal, spectral)
            x_patches = x_patches.permute(0, 2, 1, 3, 4, 5).contiguous()
            x[b] = x_patches.view(height, width, temporal, spectral)
        
        return x


def prepare_data(
    data_path: str = "./dataset",
    patch_size: int = 64,
    stride: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    准备数据加载器 - Transformer版本
    
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
    valid_mask = (y_data >= 1) & (y_data <= 8)
    valid_labels = y_data[valid_mask] - 1
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    class_weights = len(valid_labels) / (len(unique_labels) * counts)
    class_weights = torch.FloatTensor(class_weights)
    
    # 创建完整数据集用于标准化
    full_dataset = CropMappingDataset(
        x_data, y_data, 
        patch_size=patch_size,
        stride=stride,
        normalize=True,
        augment=False  # 不在这里增强
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
    
    # 创建数据集（共享标准化器）
    train_dataset = CropMappingDataset(
        x_data, y_data,
        patch_size=patch_size,
        stride=stride,
        normalize=True,
        scaler=full_dataset.scaler,
        augment=augment_train
    )
    
    val_dataset = CropMappingDataset(
        x_data, y_data,
        patch_size=patch_size,
        stride=stride,
        normalize=True,
        scaler=full_dataset.scaler,
        augment=False
    )
    
    test_dataset = CropMappingDataset(
        x_data, y_data,
        patch_size=patch_size,
        stride=stride,
        normalize=True,
        scaler=full_dataset.scaler,
        augment=False
    )
    
    # 创建数据子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
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
            0: 'Corn',
            1: 'Wheat', 
            2: 'Sunflower',
            3: 'Pumpkin',
            4: 'Artificial_Surface',
            5: 'Water',
            6: 'Road',
            7: 'Other'
        },
        'scaler': full_dataset.scaler,
        'patch_size': patch_size,
        'input_shape': (patch_size, patch_size, x_data.shape[2], x_data.shape[3]),
        'model_type': 'transformer'
    }
    
    return train_loader, val_loader, test_loader, info


def save_data_info(info: Dict[str, Any], save_path: str = "./Transformer/data_info.pkl"):
    """保存数据信息"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(info, f)
    print(f"Data info saved to {save_path}")


def load_data_info(load_path: str = "./Transformer/data_info.pkl") -> Dict[str, Any]:
    """加载数据信息"""
    with open(load_path, 'rb') as f:
        info = pickle.load(f)
    return info


if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader, info = prepare_data(
        batch_size=4,
        patch_size=64,
        augment_train=True
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class weights: {info['class_weights']}")
    print(f"Model type: {info['model_type']}")
    
    # 测试一个批次
    for batch_x, batch_y in train_loader:
        print(f"\nBatch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")
        print(f"X range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"Y unique: {torch.unique(batch_y)}")
        break