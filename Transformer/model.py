#!/usr/bin/env python3
"""
Vision Transformer Model Architecture for Crop Mapping
基于Transformer的农作物制图模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
import numpy as np


class PatchEmbedding(nn.Module):
    """
    时空切片嵌入层
    将时序多光谱数据转换为token序列
    """
    def __init__(
        self,
        patch_size: int = 8,
        input_channels: int = 8,
        temporal_steps: int = 28,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.temporal_steps = temporal_steps
        self.embed_dim = embed_dim
        
        # 时序特征提取
        self.temporal_conv = nn.Conv1d(
            input_channels, embed_dim // 2, 
            kernel_size=3, padding=1
        )
        
        # 空间切片投影
        self.spatial_projection = nn.Conv2d(
            embed_dim // 2, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x shape: (batch, height, width, temporal, spectral)
        output: (batch, num_patches, embed_dim)
        """
        batch_size, height, width, temporal, spectral = x.shape
        
        # 重塑为 (batch*height*width, spectral, temporal)
        x_reshaped = x.view(-1, temporal, spectral).transpose(1, 2)
        
        # 时序特征提取
        temporal_features = self.temporal_conv(x_reshaped)  # (batch*H*W, embed_dim//2, temporal)
        temporal_features = F.relu(temporal_features)
        
        # 时序池化
        temporal_pooled = F.adaptive_avg_pool1d(temporal_features, 1).squeeze(-1)  # (batch*H*W, embed_dim//2)
        
        # 重塑回空间维度
        temporal_pooled = temporal_pooled.view(batch_size, height, width, self.embed_dim // 2)
        temporal_pooled = temporal_pooled.permute(0, 3, 1, 2)  # (batch, embed_dim//2, height, width)
        
        # 空间切片投影
        patches = self.spatial_projection(temporal_pooled)  # (batch, embed_dim, H//patch_size, W//patch_size)
        
        # 展平为序列
        batch_size, embed_dim, patch_h, patch_w = patches.shape
        num_patches = patch_h * patch_w
        patches = patches.view(batch_size, embed_dim, num_patches).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        return self.dropout(patches)


class PositionalEncoding(nn.Module):
    """
    位置编码层
    """
    def __init__(self, embed_dim: int, max_patches: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, num_patches, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        pos_emb = self.pos_embedding[:, :num_patches, :]
        return x + pos_emb


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, num_patches, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, num_patches, head_dim)
        
        # 计算注意力分数
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)  # (batch, num_heads, num_patches, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, num_patches, embed_dim)
        """
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class CropMappingTransformer(nn.Module):
    """
    农作物制图Vision Transformer模型
    输入: (batch_size, height, width, temporal_steps, spectral_bands)
    输出: (batch_size, height, width, num_classes)
    """
    def __init__(
        self,
        input_channels: int = 8,
        temporal_steps: int = 28,
        num_classes: int = 9,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_channels = input_channels
        self.temporal_steps = temporal_steps
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 切片嵌入
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            input_channels=input_channels,
            temporal_steps=temporal_steps,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x shape: (batch_size, height, width, temporal_steps, spectral_bands)
        """
        batch_size, height, width, temporal_steps, spectral_bands = x.shape
        
        # 确保高度和宽度能被patch_size整除
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h), mode='reflect')
        
        # 切片嵌入
        patches = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # 位置编码
        patches = self.pos_encoding(patches)
        
        # Transformer编码器
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        # 层归一化
        patches = self.norm(patches)
        
        # 分类
        logits = self.classifier(patches)  # (batch, num_patches, num_classes)
        
        # 重塑回空间维度
        current_height = height + pad_h
        current_width = width + pad_w
        patch_h = current_height // self.patch_size
        patch_w = current_width // self.patch_size
        
        logits = logits.view(batch_size, patch_h, patch_w, self.num_classes)
        
        # 上采样到原始分辨率
        if patch_h != height or patch_w != width:
            logits = logits.permute(0, 3, 1, 2)  # (batch, num_classes, patch_h, patch_w)
            logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
            logits = logits.permute(0, 2, 3, 1)  # (batch, height, width, num_classes)
        
        return logits


def create_transformer_model(
    input_channels: int = 8,
    temporal_steps: int = 28,
    num_classes: int = 9,
    patch_size: int = 8,
    embed_dim: int = 256,
    num_layers: int = 6,
    **kwargs
) -> CropMappingTransformer:
    """
    创建Transformer模型的工厂函数
    """
    model = CropMappingTransformer(
        input_channels=input_channels,
        temporal_steps=temporal_steps,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_transformer_model()
    print(f"Model architecture:\n{model}")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 64, 64, 28, 8)  # batch=2, spatial=64x64
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")