#!/usr/bin/env python3
"""
TCN Model Architecture for Crop Mapping
Temporal Convolutional Network implementation for multi-spectral time series classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CausalConv1d(nn.Module):
    """
    因果卷积层，确保时序数据的因果性
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, sequence)
        x = self.conv(x)
        if self.padding > 0:
            return x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """
    TCN基本块，包含两个因果卷积层和残差连接
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            n_inputs, n_outputs, kernel_size, dilation=dilation
        )
        self.conv2 = CausalConv1d(
            n_outputs, n_outputs, kernel_size, dilation=dilation
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积块
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 残差连接
        residual = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    时序卷积网络主体
    """
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CropMappingTCN(nn.Module):
    """
    农作物制图TCN模型
    输入: (batch_size, height, width, temporal_steps, spectral_bands)
    输出: (batch_size, height, width, num_classes)
    """
    def __init__(
        self,
        input_channels: int = 8,  # 光谱波段数
        temporal_steps: int = 28,  # 时间步数
        num_classes: int = 9,  # 类别数
        tcn_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_channels = input_channels
        self.temporal_steps = temporal_steps
        self.num_classes = num_classes
        
        # TCN层
        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 全局池化和分类层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x shape: (batch_size, height, width, temporal_steps, spectral_bands)
        """
        batch_size, height, width, temporal_steps, spectral_bands = x.shape
        
        # 重塑数据以适配TCN输入
        # (batch*height*width, spectral_bands, temporal_steps)
        x = x.view(-1, temporal_steps, spectral_bands)
        x = x.transpose(1, 2)  # (batch*height*width, spectral_bands, temporal_steps)
        
        # 通过TCN
        x = self.tcn(x)  # (batch*height*width, tcn_channels[-1], temporal_steps)
        
        # 全局池化
        x = self.global_pool(x)  # (batch*height*width, tcn_channels[-1], 1)
        x = x.squeeze(-1)  # (batch*height*width, tcn_channels[-1])
        
        # 分类
        x = self.classifier(x)  # (batch*height*width, num_classes)
        
        # 重塑回原始空间维度
        x = x.view(batch_size, height, width, self.num_classes)
        
        return x


def create_tcn_model(
    input_channels: int = 8,
    temporal_steps: int = 28,
    num_classes: int = 9,
    tcn_channels: List[int] = None,
    **kwargs
) -> CropMappingTCN:
    """
    创建TCN模型的工厂函数
    """
    if tcn_channels is None:
        tcn_channels = [64, 128, 256]
    
    model = CropMappingTCN(
        input_channels=input_channels,
        temporal_steps=temporal_steps,
        num_classes=num_classes,
        tcn_channels=tcn_channels,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_tcn_model()
    print(f"Model architecture:\n{model}")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 32, 32, 28, 8)  # batch=2, spatial=32x32
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")