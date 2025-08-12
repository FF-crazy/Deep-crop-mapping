#!/usr/bin/env python3
"""
推理脚本 - Transformer版本
使用训练好的Transformer模型进行农作物制图预测
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

from Transformer.model import create_transformer_model
from Transformer.dataset import load_data_info
from Transformer.utils import load_model


class CropMappingTransformerInference:
    """
    农作物制图Transformer推理类
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
        self.model = create_transformer_model(
            input_channels=self.config.get('input_channels', 8),
            temporal_steps=self.config.get('temporal_steps', 28),
            num_classes=self.data_info['num_classes'],
            patch_size=self.config.get('model_patch_size', 8),
            embed_dim=self.config.get('embed_dim', 256),
            num_layers=self.config.get('num_layers', 6),
            num_heads=self.config.get('num_heads', 8),
            mlp_ratio=self.config.get('mlp_ratio', 4.0),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载数据标准化器
        self.scaler = self.data_info.get('scaler', None)
        
        print(f"Transformer model loaded successfully on {self.device}")
        print(f"Number of classes: {self.data_info['num_classes']}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
    
    def predict_patch(self, patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测单个切片
        
        Args:
            patch: 输入切片 (height, width, temporal, spectral)
            
        Returns:
            predictions: 预测结果 (height, width)
            confidence: 置信度 (height, width)
        """
        with torch.no_grad():
            # 添加批次维度
            patch = patch.unsqueeze(0).to(self.device)
            
            # 预测
            outputs = self.model(patch)  # (1, height, width, num_classes)
            probs = torch.softmax(outputs, dim=-1)
            confidence, predictions = probs.max(-1)
            
            return predictions.squeeze(0).cpu(), confidence.squeeze(0).cpu()
    
    def predict_full_image(
        self, 
        x_data: np.ndarray,
        patch_size: int = 64,
        overlap: int = 16,
        batch_size: int = 4,
        use_tta: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测完整图像（使用滑动窗口）
        
        Args:
            x_data: 完整输入数据 (height, width, temporal, spectral)
            patch_size: 切片大小
            overlap: 重叠大小
            batch_size: 批处理大小
            use_tta: 是否使用测试时间增强
            
        Returns:
            predictions: 预测结果 (height, width)
            confidence_map: 置信度图 (height, width)
        """
        height, width = x_data.shape[:2]
        stride = patch_size - overlap
        
        # 创建输出数组
        predictions_sum = np.zeros((height, width, self.data_info['num_classes']), dtype=np.float32)
        confidence_sum = np.zeros((height, width), dtype=np.float32)
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
        
        # 处理边界情况
        if height % stride != 0:
            for j in range(0, width - patch_size + 1, stride):
                i = height - patch_size
                patch = x_tensor[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                positions.append((i, j))
        
        if width % stride != 0:
            for i in range(0, height - patch_size + 1, stride):
                j = width - patch_size
                patch = x_tensor[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                positions.append((i, j))
        
        # 处理右下角
        if height % stride != 0 and width % stride != 0:
            i = height - patch_size
            j = width - patch_size
            patch = x_tensor[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))
        
        print(f"Predicting {len(patches)} patches with Transformer...")
        
        # 批量预测
        for batch_start in range(0, len(patches), batch_size):
            batch_end = min(batch_start + batch_size, len(patches))
            batch_patches = torch.stack(patches[batch_start:batch_end])
            batch_positions = positions[batch_start:batch_end]
            
            # TTA (测试时间增强)
            if use_tta:
                batch_probs = self._predict_with_tta(batch_patches)
            else:
                # 标准预测
                with torch.no_grad():
                    batch_patches = batch_patches.to(self.device)
                    outputs = self.model(batch_patches)
                    batch_probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            
            # 累积预测结果
            for probs, (i, j) in zip(batch_probs, batch_positions):
                predictions_sum[i:i+patch_size, j:j+patch_size] += probs
                confidence_sum[i:i+patch_size, j:j+patch_size] += probs.max(axis=-1)
                counts[i:i+patch_size, j:j+patch_size] += 1
        
        # 平均重叠区域的预测
        predictions_avg = predictions_sum / (counts[:, :, np.newaxis] + 1e-8)
        confidence_avg = confidence_sum / (counts + 1e-8)
        
        # 获取最终预测
        final_predictions = np.argmax(predictions_avg, axis=-1)
        
        return final_predictions, confidence_avg
    
    def _predict_with_tta(self, batch_patches: torch.Tensor) -> np.ndarray:
        """
        使用测试时间增强进行预测
        """
        all_probs = []
        
        with torch.no_grad():
            batch_patches_device = batch_patches.to(self.device)
            
            # 原始预测
            outputs = self.model(batch_patches_device)
            probs = torch.softmax(outputs, dim=-1)
            all_probs.append(probs.cpu().numpy())
            
            # 水平翻转
            flipped_h = torch.flip(batch_patches_device, dims=[2])
            outputs_h = self.model(flipped_h)
            probs_h = torch.softmax(outputs_h, dim=-1)
            probs_h = torch.flip(probs_h, dims=[2])
            all_probs.append(probs_h.cpu().numpy())
            
            # 垂直翻转
            flipped_v = torch.flip(batch_patches_device, dims=[1])
            outputs_v = self.model(flipped_v)
            probs_v = torch.softmax(outputs_v, dim=-1)
            probs_v = torch.flip(probs_v, dims=[1])
            all_probs.append(probs_v.cpu().numpy())
            
            # 旋转90度
            rotated = torch.rot90(batch_patches_device, k=1, dims=[1, 2])
            outputs_r = self.model(rotated)
            probs_r = torch.softmax(outputs_r, dim=-1)
            probs_r = torch.rot90(probs_r, k=-1, dims=[1, 2])
            all_probs.append(probs_r.cpu().numpy())
        
        # 平均所有预测
        avg_probs = np.mean(all_probs, axis=0)
        
        return avg_probs
    
    def visualize_results(
        self,
        predictions: np.ndarray,
        confidence_map: np.ndarray = None,
        save_path: Optional[str] = None,
        show_legend: bool = True,
        show_confidence: bool = True
    ):
        """
        可视化预测结果 - Transformer版本
        """
        if show_confidence and confidence_map is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        
        # 创建颜色映射
        colors = [
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
        im1 = ax1.imshow(predictions, cmap=cmap, norm=norm)
        ax1.set_title('Transformer Crop Type Predictions', fontsize=16)
        ax1.axis('off')
        
        # 添加图例
        if show_legend:
            from matplotlib.patches import Patch
            legend_elements = []
            for i, class_name in self.data_info['class_names'].items():
                if i < len(colors):
                    legend_elements.append(
                        Patch(facecolor=colors[i], label=class_name)
                    )
            ax1.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1, 0.5), fontsize=12)
        
        # 绘制置信度图
        if show_confidence and confidence_map is not None:
            im2 = ax2.imshow(confidence_map, cmap='viridis')
            ax2.set_title('Prediction Confidence', fontsize=16)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Confidence')
        
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
    parser = argparse.ArgumentParser(description='Run inference with Transformer model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data (x.npy)')
    parser.add_argument('--output-dir', type=str, default='./Transformer/predictions', help='Output directory')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size for inference')
    parser.add_argument('--overlap', type=int, default=16, help='Overlap between patches')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--use-tta', action='store_true', help='Use test time augmentation')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--save-geotiff', action='store_true', help='Save as GeoTIFF')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化推理器
    inferencer = CropMappingTransformerInference(args.model_path)
    
    # 加载输入数据
    print(f"Loading data from {args.input_path}...")
    x_data = np.load(args.input_path)
    print(f"Input shape: {x_data.shape}")
    
    # 运行推理
    start_time = time.time()
    predictions, confidence_map = inferencer.predict_full_image(
        x_data,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        use_tta=args.use_tta
    )
    inference_time = time.time() - start_time
    
    print(f"Transformer inference completed in {inference_time:.2f} seconds")
    print(f"Predictions shape: {predictions.shape}")
    
    # 保存预测结果
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'confidence_map.npy', confidence_map)
    
    # 计算预测统计
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction statistics:")
    for label, count in zip(unique, counts):
        percentage = count / predictions.size * 100
        class_name = inferencer.data_info['class_names'].get(label, f"Unknown({label})")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
    
    # 统计置信度
    print(f"\nConfidence statistics:")
    print(f"  Mean confidence: {confidence_map.mean():.4f}")
    print(f"  Min confidence: {confidence_map.min():.4f}")
    print(f"  Max confidence: {confidence_map.max():.4f}")
    
    # 可视化结果
    if args.visualize:
        inferencer.visualize_results(
            predictions,
            confidence_map,
            save_path=output_dir / 'transformer_predictions_visualization.png',
            show_confidence=True
        )
    
    # 保存为GeoTIFF
    if args.save_geotiff:
        inferencer.save_geotiff(
            predictions,
            save_path=str(output_dir / 'predictions.tif')
        )
        
        # 也保存置信度图
        inferencer.save_geotiff(
            confidence_map,
            save_path=str(output_dir / 'confidence.tif')
        )
    
    print(f"\nTransformer inference results saved to {output_dir}")


if __name__ == "__main__":
    main()