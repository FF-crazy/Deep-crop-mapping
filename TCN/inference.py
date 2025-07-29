#!/usr/bin/env python3
"""
推理脚本
使用训练好的TCN模型进行农作物制图预测
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Any, Tuple, Optional
import time
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from TCN.model import create_tcn_model
from TCN.dataset import load_data_info
from TCN.utils import load_model


class CropMappingInference:
    """
    农作物制图推理类
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型配置
        from .compat import load_checkpoint
        checkpoint = load_checkpoint(model_path, map_location=str(self.device))
        self.config = checkpoint.get('config', {})
        
        # 加载数据信息
        model_dir = Path(model_path).parent
        self.data_info = load_data_info(f"{model_dir}/data_info.pkl")
        
        # 创建模型
        self.model = create_tcn_model(
            input_channels=self.config.get('input_channels', 8),
            temporal_steps=self.config.get('temporal_steps', 28),
            num_classes=self.data_info['num_classes'],
            tcn_channels=self.config.get('tcn_channels', [64, 128, 256]),
            kernel_size=self.config.get('kernel_size', 3),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载数据标准化器
        self.scaler = self.data_info.get('scaler', None)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Number of classes: {self.data_info['num_classes']}")
        
    def preprocess_data(self, x_data: np.ndarray) -> torch.Tensor:
        """
        预处理输入数据
        
        Args:
            x_data: 输入数据 (height, width, temporal, spectral)
            
        Returns:
            处理后的张量
        """
        # 标准化
        if self.scaler is not None:
            height, width, temporal, spectral = x_data.shape
            x_reshaped = x_data.reshape(-1, spectral)
            x_normalized = self.scaler.transform(x_reshaped)
            x_data = x_normalized.reshape(height, width, temporal, spectral)
        
        # 转换为张量
        x_tensor = torch.from_numpy(x_data).float()
        
        return x_tensor
    
    def predict_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        预测单个切片
        
        Args:
            patch: 输入切片 (height, width, temporal, spectral)
            
        Returns:
            预测结果 (height, width)
        """
        with torch.no_grad():
            # 添加批次维度
            patch = patch.unsqueeze(0).to(self.device)
            
            # 预测
            outputs = self.model(patch)
            _, predictions = outputs.max(-1)
            
            return predictions.squeeze(0).cpu()
    
    def predict_full_image(
        self, 
        x_data: np.ndarray,
        patch_size: int = 64,
        overlap: int = 16,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        预测完整图像（使用滑动窗口）
        
        Args:
            x_data: 完整输入数据 (height, width, temporal, spectral)
            patch_size: 切片大小
            overlap: 重叠大小
            batch_size: 批处理大小
            
        Returns:
            预测结果 (height, width)
        """
        height, width = x_data.shape[:2]
        stride = patch_size - overlap
        
        # 创建输出数组和计数数组
        predictions = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        
        # 预处理数据
        x_tensor = self.preprocess_data(x_data)
        
        # 收集所有切片
        patches = []
        positions = []
        
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = x_tensor[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                positions.append((i, j))
        
        # 批量预测
        print(f"Predicting {len(patches)} patches...")
        
        for batch_start in range(0, len(patches), batch_size):
            batch_end = min(batch_start + batch_size, len(patches))
            batch_patches = torch.stack(patches[batch_start:batch_end])
            batch_positions = positions[batch_start:batch_end]
            
            # 批量预测
            with torch.no_grad():
                batch_patches = batch_patches.to(self.device)
                outputs = self.model(batch_patches)
                _, batch_predictions = outputs.max(-1)
                batch_predictions = batch_predictions.cpu().numpy()
            
            # 累积预测结果
            for pred, (i, j) in zip(batch_predictions, batch_positions):
                predictions[i:i+patch_size, j:j+patch_size] += pred
                counts[i:i+patch_size, j:j+patch_size] += 1
        
        # 平均重叠区域的预测
        predictions = (predictions / (counts + 1e-8)).astype(np.int32)
        
        return predictions
    
    def visualize_results(
        self,
        predictions: np.ndarray,
        save_path: Optional[str] = None,
        show_legend: bool = True
    ):
        """
        可视化预测结果
        """
        plt.figure(figsize=(12, 10))
        
        # 创建颜色映射
        colors = [
            '#000000',  # Background
            '#FFD700',  # Corn
            '#8B4513',  # Wheat
            '#FFA500',  # Sunflower
            '#FF6347',  # Pumpkin
            '#808080',  # Artificial Surface
            '#0000FF',  # Water
            '#696969',  # Road
            '#9ACD32'   # Other
        ]
        
        cmap = plt.cm.colors.ListedColormap(colors[:self.data_info['num_classes']])
        bounds = list(range(self.data_info['num_classes'] + 1))
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # 绘制预测图
        im = plt.imshow(predictions, cmap=cmap, norm=norm)
        plt.title('Crop Type Predictions', fontsize=16)
        plt.axis('off')
        
        # 添加图例
        if show_legend:
            from matplotlib.patches import Patch
            legend_elements = []
            for i, class_name in self.data_info['class_names'].items():
                if i < len(colors):
                    legend_elements.append(
                        Patch(facecolor=colors[i], label=class_name)
                    )
            plt.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1, 0.5), fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_geotiff(
        self,
        predictions: np.ndarray,
        save_path: str,
        transform: Optional[Any] = None,
        crs: Optional[str] = None
    ):
        """
        保存预测结果为GeoTIFF格式
        """
        height, width = predictions.shape
        
        # 如果没有提供变换矩阵，创建默认的
        if transform is None:
            transform = from_bounds(0, 0, width, height, width, height)
        
        # 默认坐标系
        if crs is None:
            crs = 'EPSG:4326'
        
        # 写入GeoTIFF
        with rasterio.open(
            save_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=predictions.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(predictions, 1)
        
        print(f"Predictions saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with TCN model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data (x.npy)')
    parser.add_argument('--output-dir', type=str, default='./TCN/predictions', help='Output directory')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size for inference')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between patches')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--save-geotiff', action='store_true', help='Save as GeoTIFF')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化推理器
    inferencer = CropMappingInference(args.model_path)
    
    # 加载输入数据
    print(f"Loading data from {args.input_path}...")
    x_data = np.load(args.input_path)
    print(f"Input shape: {x_data.shape}")
    
    # 运行推理
    start_time = time.time()
    predictions = inferencer.predict_full_image(
        x_data,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size
    )
    inference_time = time.time() - start_time
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Predictions shape: {predictions.shape}")
    
    # 保存预测结果
    np.save(output_dir / 'predictions.npy', predictions)
    
    # 计算预测统计
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction statistics:")
    for label, count in zip(unique, counts):
        percentage = count / predictions.size * 100
        class_name = inferencer.data_info['class_names'].get(label, f"Unknown({label})")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
    
    # 可视化结果
    if args.visualize:
        inferencer.visualize_results(
            predictions,
            save_path=output_dir / 'predictions_visualization.png'
        )
    
    # 保存为GeoTIFF
    if args.save_geotiff:
        inferencer.save_geotiff(
            predictions,
            save_path=str(output_dir / 'predictions.tif')
        )
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()