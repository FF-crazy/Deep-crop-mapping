#!/usr/bin/env python3
"""
数据集可视化脚本
用于分析和可视化x.npy和y.npy数据集，帮助进行数据理解和决策

Author: DeepCropMapping Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import argparse
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class CropDataVisualizer:
    """农作物数据可视化器"""
    
    def __init__(self, data_dir: str = "../dataset"):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = Path(data_dir)
        self.x_data = None
        self.y_data = None
        
        # 作物类别定义
        self.crop_labels = {
            0: 'Background/Unknown',
            1: 'Corn (玉米)',
            2: 'Wheat (小麦)', 
            3: 'Sunflower (向日葵)',
            4: 'Pumpkin (番瓜)',
            5: 'Artificial Surface (人造地表)',
            6: 'Water (水体)',
            7: 'Road (道路)',
            8: 'Other (其他)'
        }
        
        # 颜色映射
        self.colors = [
            '#000000',  # 0: 黑色 - Background
            '#FFD700',  # 1: 金色 - 玉米
            '#8B4513',  # 2: 棕色 - 小麦
            '#FFA500',  # 3: 橙色 - 向日葵  
            '#FF6347',  # 4: 番茄色 - 番瓜
            '#808080',  # 5: 灰色 - 人造地表
            '#0000FF',  # 6: 蓝色 - 水体
            '#696969',  # 7: 深灰 - 道路
            '#9ACD32'   # 8: 黄绿色 - 其他
        ]
        
        self.load_data()
    
    def load_data(self) -> None:
        """加载数据集"""
        try:
            print("Loading dataset...")
            x_path = self.data_dir / "x.npy"
            y_path = self.data_dir / "y.npy"
            
            if not x_path.exists() or not y_path.exists():
                raise FileNotFoundError(f"Data files not found in {self.data_dir}")
            
            self.x_data = np.load(x_path)  # (326, 1025, 28, 8)
            self.y_data = np.load(y_path)  # (326, 1025)
            
            print(f"X data shape: {self.x_data.shape}")
            print(f"Y data shape: {self.y_data.shape}")
            print("Dataset loaded successfully!\n")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def data_overview(self) -> None:
        """数据概览"""
        print("=" * 50)
        print("DATASET OVERVIEW")
        print("=" * 50)
        
        # X数据统计
        print("Multi-spectral Data (X):")
        print(f"  Shape: {self.x_data.shape}")
        print(f"  Data type: {self.x_data.dtype}")
        print(f"  Memory usage: {self.x_data.nbytes / (1024**2):.2f} MB")
        print(f"  Value range: [{self.x_data.min()}, {self.x_data.max()}]")
        print(f"  Mean value: {self.x_data.mean():.2f}")
        print(f"  Std deviation: {self.x_data.std():.2f}")
        
        # Y数据统计
        print(f"\nLabel Data (Y):")
        print(f"  Shape: {self.y_data.shape}")
        print(f"  Data type: {self.y_data.dtype}")
        print(f"  Unique labels: {np.unique(self.y_data)}")
        
        # 各类别统计
        unique_labels, counts = np.unique(self.y_data, return_counts=True)
        total_pixels = self.y_data.size
        
        print(f"\nClass Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = count / total_pixels * 100
            crop_name = self.crop_labels.get(label, f"Unknown({label})")
            print(f"  {crop_name}: {count:,} pixels ({percentage:.2f}%)")
    
    def plot_class_distribution(self, save_path: Optional[str] = None) -> None:
        """绘制类别分布图"""
        unique_labels, counts = np.unique(self.y_data, return_counts=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 饼图
        crop_names = [self.crop_labels.get(label, f"Class {label}") for label in unique_labels]
        colors_subset = [self.colors[label] for label in unique_labels]
        
        wedges, texts, autotexts = ax1.pie(counts, labels=crop_names, colors=colors_subset, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Crop Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        
        # 条形图
        bars = ax2.bar(range(len(unique_labels)), counts, color=colors_subset)
        ax2.set_xlabel('Crop Classes', fontweight='bold')
        ax2.set_ylabel('Number of Pixels', fontweight='bold')
        ax2.set_title('Crop Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(unique_labels)))
        ax2.set_xticklabels([f"{label}\n{name.split('(')[0].strip()}" 
                            for label, name in zip(unique_labels, crop_names)], 
                           rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_spatial_distribution(self, save_path: Optional[str] = None) -> None:
        """绘制空间分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 创建颜色映射
        cmap = mcolors.ListedColormap(self.colors[:len(np.unique(self.y_data))])
        bounds = list(range(len(np.unique(self.y_data)) + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 完整标签图
        im1 = axes[0,0].imshow(self.y_data, cmap=cmap, norm=norm)
        axes[0,0].set_title('Complete Label Map', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Width (pixels)')
        axes[0,0].set_ylabel('Height (pixels)')
        
        # 局部放大图 (中心区域)
        h, w = self.y_data.shape
        center_h, center_w = h//2, w//2
        crop_size = 200
        crop_data = self.y_data[center_h-crop_size//2:center_h+crop_size//2,
                               center_w-crop_size//2:center_w+crop_size//2]
        
        im2 = axes[0,1].imshow(crop_data, cmap=cmap, norm=norm)
        axes[0,1].set_title('Center Region (200x200)', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Width (pixels)')
        axes[0,1].set_ylabel('Height (pixels)')
        
        # 按行统计
        row_distribution = np.array([np.bincount(self.y_data[i], minlength=9) 
                                    for i in range(self.y_data.shape[0])])
        im3 = axes[1,0].imshow(row_distribution.T, aspect='auto', cmap='viridis')
        axes[1,0].set_title('Distribution by Rows', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Row Index')
        axes[1,0].set_ylabel('Crop Class')
        
        # 按列统计  
        col_distribution = np.array([np.bincount(self.y_data[:, j], minlength=9) 
                                    for j in range(self.y_data.shape[1])])
        im4 = axes[1,1].imshow(col_distribution.T, aspect='auto', cmap='viridis')
        axes[1,1].set_title('Distribution by Columns', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Column Index')
        axes[1,1].set_ylabel('Crop Class')
        
        # 添加颜色条
        unique_labels = np.unique(self.y_data)
        cbar = plt.colorbar(im1, ax=axes[0,:], orientation='horizontal', 
                           fraction=0.05, pad=0.1, shrink=0.8)
        cbar.set_ticks(unique_labels)
        cbar.set_ticklabels([self.crop_labels[label].split('(')[0].strip() 
                            for label in unique_labels])
        cbar.set_label('Crop Classes', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spatial distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_spectral_analysis(self, sample_points: int = 100, 
                              save_path: Optional[str] = None) -> None:
        """光谱分析图"""
        # 随机采样点进行分析
        h, w = self.y_data.shape
        sample_indices = np.random.choice(h * w, sample_points, replace=False)
        sample_coords = [(idx // w, idx % w) for idx in sample_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 各波段均值分布
        band_means = np.mean(self.x_data, axis=(0, 1, 2))  # 对空间和时间维度求均值
        axes[0,0].bar(range(8), band_means, color='skyblue', edgecolor='navy')
        axes[0,0].set_title('Mean Values Across Bands', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Band Index')
        axes[0,0].set_ylabel('Mean Value')
        axes[0,0].set_xticks(range(8))
        
        # 各波段方差分布
        band_stds = np.std(self.x_data, axis=(0, 1, 2))
        axes[0,1].bar(range(8), band_stds, color='lightcoral', edgecolor='darkred')
        axes[0,1].set_title('Standard Deviation Across Bands', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Band Index')
        axes[0,1].set_ylabel('Standard Deviation')
        axes[0,1].set_xticks(range(8))
        
        # 不同类别的光谱特征对比 (第一期数据)
        unique_labels = np.unique(self.y_data)
        for label in unique_labels[:6]:  # 只显示前6个类别避免图像过乱
            mask = self.y_data == label
            if np.sum(mask) > 10:  # 确保有足够的样本
                label_data = self.x_data[mask, 0, :]  # 取第一期数据
                mean_spectrum = np.mean(label_data, axis=0)
                axes[1,0].plot(range(8), mean_spectrum, marker='o', 
                             label=self.crop_labels[label].split('(')[0].strip(),
                             linewidth=2)
        
        axes[1,0].set_title('Spectral Signatures by Crop Type (First Period)', 
                           fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Band Index')
        axes[1,0].set_ylabel('Mean Spectral Value')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 光谱值分布直方图
        random_pixels = self.x_data.reshape(-1, 28, 8)[::1000]  # 每1000个像素取1个
        all_bands = random_pixels[:, 0, :].flatten()  # 取第一期的所有波段
        axes[1,1].hist(all_bands, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
        axes[1,1].set_title('Spectral Value Distribution', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Spectral Value')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectral analysis plot saved to: {save_path}")
        
        plt.show()
    
    def plot_temporal_analysis(self, sample_pixels: int = 20, 
                              save_path: Optional[str] = None) -> None:
        """时序数据分析"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. 不同作物类型的NDVI时序曲线
        unique_labels = np.unique(self.y_data)
        
        # 计算NDVI (假设Band4=NIR, Band3=Red)
        nir = self.x_data[:, :, :, 3].astype(np.float32)
        red = self.x_data[:, :, :, 2].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        ax = axes[0,0]
        for label in unique_labels[:6]:  # 只显示前6个类别
            mask = self.y_data == label
            if np.sum(mask) > 10:
                label_ndvi = ndvi[mask]  # (num_pixels, 28)
                mean_ndvi = np.mean(label_ndvi, axis=0)
                std_ndvi = np.std(label_ndvi, axis=0)
                
                time_steps = range(28)
                ax.plot(time_steps, mean_ndvi, marker='o', linewidth=2,
                       label=self.crop_labels[label].split('(')[0].strip())
                ax.fill_between(time_steps, mean_ndvi-std_ndvi, mean_ndvi+std_ndvi, alpha=0.2)
        
        ax.set_title('NDVI Time Series by Crop Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period (Apr-Oct 2021)')
        ax.set_ylabel('NDVI Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. 随机像素点的光谱时序变化
        ax = axes[0,1]
        h, w = self.y_data.shape
        for i in range(min(sample_pixels, 10)):
            rand_h = np.random.randint(0, h)
            rand_w = np.random.randint(0, w)
            pixel_data = self.x_data[rand_h, rand_w, :, 0]  # 选择第一个波段
            ax.plot(range(28), pixel_data, alpha=0.7, linewidth=1)
        
        ax.set_title(f'Band 1 Temporal Variation ({min(sample_pixels, 10)} Random Pixels)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Spectral Value')
        ax.grid(True, alpha=0.3)
        
        # 3. 各时期的数据统计
        ax = axes[1,0]
        temporal_means = np.mean(self.x_data, axis=(0, 1, 3))  # 对空间和波段维度求均值
        temporal_stds = np.std(self.x_data, axis=(0, 1, 3))
        
        ax.plot(range(28), temporal_means, 'b-o', label='Mean', linewidth=2)
        ax.fill_between(range(28), temporal_means-temporal_stds, temporal_means+temporal_stds, 
                       alpha=0.3, label='±1 Std')
        ax.set_title('Overall Spectral Statistics Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Spectral Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 时序数据的热力图
        ax = axes[1,1]
        # 随机选择一些像素进行可视化
        sample_indices = np.random.choice(h * w, 50, replace=False)
        sample_coords = [(idx // w, idx % w) for idx in sample_indices]
        
        temporal_matrix = []
        for coord in sample_coords[:20]:  # 只显示20个像素
            pixel_temporal = self.x_data[coord[0], coord[1], :, 0]  # Band 1
            temporal_matrix.append(pixel_temporal)
        
        temporal_matrix = np.array(temporal_matrix)
        im = ax.imshow(temporal_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Temporal Heatmap (Band 1, 20 Random Pixels)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Pixel Index')
        plt.colorbar(im, ax=ax)
        
        # 5. 季节性变化分析
        ax = axes[2,0]
        # 计算各月份的平均NDVI
        monthly_ndvi = []
        months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        periods_per_month = 28 // 7  # 每月大约4期
        
        for month_idx in range(7):
            start_period = month_idx * periods_per_month
            end_period = min((month_idx + 1) * periods_per_month, 28)
            if start_period < 28:
                month_ndvi = np.mean(ndvi[:, :, start_period:end_period])
                monthly_ndvi.append(month_ndvi)
            else:
                monthly_ndvi.append(monthly_ndvi[-1])  # 最后一个月复制上一个值
        
        ax.bar(months[:len(monthly_ndvi)], monthly_ndvi, color='green', alpha=0.7)
        ax.set_title('Monthly Average NDVI', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('NDVI Value')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. 时序数据的统计分布
        ax = axes[2,1]
        # 计算每个像素的时序方差
        pixel_variance = np.var(self.x_data[:, :, :, 0], axis=2)  # 时间维度的方差
        
        ax.hist(pixel_variance.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.set_title('Distribution of Temporal Variance (Band 1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Temporal Variance')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal analysis plot saved to: {save_path}")
        
        plt.show()
    
    def interactive_pixel_explorer(self, x_coord: int = None, y_coord: int = None) -> None:
        """交互式像素点探索"""
        h, w = self.y_data.shape
        
        if x_coord is None or y_coord is None:
            # 随机选择一个像素点
            x_coord = np.random.randint(0, w)
            y_coord = np.random.randint(0, h)
        
        # 确保坐标在有效范围内
        x_coord = max(0, min(x_coord, w-1))
        y_coord = max(0, min(y_coord, h-1))
        
        # 获取该像素点的数据
        pixel_label = self.y_data[y_coord, x_coord]
        pixel_spectral = self.x_data[y_coord, x_coord, :, :]  # (28, 8)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 该像素点的所有波段时序曲线
        ax = axes[0,0]
        for band in range(8):
            ax.plot(range(28), pixel_spectral[:, band], marker='o', 
                   label=f'Band {band+1}', linewidth=2)
        
        ax.set_title(f'Pixel ({y_coord}, {x_coord}) - All Bands Time Series\n'
                    f'Label: {pixel_label} ({self.crop_labels.get(pixel_label, "Unknown")})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Spectral Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. NDVI时序曲线
        ax = axes[0,1]
        nir = pixel_spectral[:, 3].astype(np.float32)
        red = pixel_spectral[:, 2].astype(np.float32)
        pixel_ndvi = (nir - red) / (nir + red + 1e-8)
        
        ax.plot(range(28), pixel_ndvi, 'g-o', linewidth=3, markersize=6)
        ax.set_title(f'Pixel ({y_coord}, {x_coord}) - NDVI Time Series', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('NDVI Value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Vegetation Threshold')
        ax.legend()
        
        # 3. 光谱特征热力图
        ax = axes[1,0]
        im = ax.imshow(pixel_spectral.T, cmap='viridis', aspect='auto')
        ax.set_title(f'Spectral-Temporal Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Band Index')
        ax.set_yticks(range(8))
        ax.set_yticklabels([f'Band {i+1}' for i in range(8)])
        plt.colorbar(im, ax=ax)
        
        # 4. 周围区域的标签分布
        ax = axes[1,1]
        # 提取该像素点周围的小区域
        patch_size = 20
        start_y = max(0, y_coord - patch_size//2)
        end_y = min(h, y_coord + patch_size//2)
        start_x = max(0, x_coord - patch_size//2)
        end_x = min(w, x_coord + patch_size//2)
        
        local_patch = self.y_data[start_y:end_y, start_x:end_x]
        
        # 创建颜色映射
        unique_local = np.unique(local_patch)
        cmap = mcolors.ListedColormap([self.colors[i] for i in unique_local])
        bounds = list(range(len(unique_local) + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        im = ax.imshow(local_patch, cmap=cmap, norm=norm)
        
        # 标记目标像素点
        target_y = y_coord - start_y
        target_x = x_coord - start_x  
        ax.plot(target_x, target_y, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='white')
        
        ax.set_title(f'Local Region Around Pixel ({y_coord}, {x_coord})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate (relative)')
        ax.set_ylabel('Y Coordinate (relative)')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细信息
        print(f"\n=== Pixel Information ===")
        print(f"Coordinates: ({y_coord}, {x_coord})")
        print(f"Label: {pixel_label} - {self.crop_labels.get(pixel_label, 'Unknown')}")
        print(f"Spectral range: [{pixel_spectral.min():.2f}, {pixel_spectral.max():.2f}]")
        print(f"NDVI range: [{pixel_ndvi.min():.3f}, {pixel_ndvi.max():.3f}]")
        print(f"NDVI mean: {pixel_ndvi.mean():.3f}")
    
    def generate_report(self, output_dir: str = "./visualization_output") -> None:
        """生成完整的数据分析报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive data analysis report...")
        
        # 生成所有可视化图表
        self.plot_class_distribution(f"{output_dir}/class_distribution.png")
        self.plot_spatial_distribution(f"{output_dir}/spatial_distribution.png")
        self.plot_spectral_analysis(f"{output_dir}/spectral_analysis.png")
        self.plot_temporal_analysis(f"{output_dir}/temporal_analysis.png")
        
        # 生成文本报告
        report_path = f"{output_dir}/data_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DEEPCROPMAPPING DATASET ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # 数据概况
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Multi-spectral data shape: {self.x_data.shape}\n")
            f.write(f"Label data shape: {self.y_data.shape}\n")
            f.write(f"Data type: {self.x_data.dtype}\n")
            f.write(f"Memory usage: {self.x_data.nbytes / (1024**2):.2f} MB\n\n")
            
            # 类别分布
            unique_labels, counts = np.unique(self.y_data, return_counts=True)
            total_pixels = self.y_data.size
            
            f.write("2. CLASS DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            for label, count in zip(unique_labels, counts):
                percentage = count / total_pixels * 100
                crop_name = self.crop_labels.get(label, f"Unknown({label})")
                f.write(f"{crop_name}: {count:,} pixels ({percentage:.2f}%)\n")
            
            # 数据质量评估
            f.write("\n3. DATA QUALITY ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            
            # 检查异常值
            extreme_high = np.sum(self.x_data > np.percentile(self.x_data, 99.9))
            extreme_low = np.sum(self.x_data == 0)
            f.write(f"Extreme high values (>99.9 percentile): {extreme_high:,}\n")
            f.write(f"Zero values: {extreme_low:,}\n")
            
            # 时序数据连续性
            temporal_mean = np.mean(self.x_data, axis=(0, 1, 3))
            temporal_diff = np.diff(temporal_mean)
            max_jump = np.max(np.abs(temporal_diff))
            f.write(f"Maximum temporal jump: {max_jump:.2f}\n")
            
            f.write("\n4. RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("Based on the analysis, consider the following:\n")
            f.write("- Apply data normalization to handle value range differences\n")
            f.write("- Check for class imbalance and consider resampling strategies\n")
            f.write("- Remove or interpolate extreme outlier values\n")
            f.write("- Apply temporal smoothing for noisy time series data\n")
        
        print(f"Report generated successfully in: {output_dir}/")
        print("Generated files:")
        print("  - class_distribution.png")
        print("  - spatial_distribution.png") 
        print("  - spectral_analysis.png")
        print("  - temporal_analysis.png")
        print("  - data_analysis_report.txt")


def main():
    """主函数 - 命令行接口"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../dataset"
    
    # 初始化可视化器
    visualizer = CropDataVisualizer(data_dir)
    
    # 显示数据概览
    visualizer.data_overview()
    
    print("\nStarting visualization analysis...")
    print("Close each plot window to proceed to the next visualization.")
    
    # 依次展示各种可视化
    try:
        visualizer.plot_class_distribution()
        visualizer.plot_spatial_distribution()
        visualizer.plot_spectral_analysis()
        visualizer.plot_temporal_analysis()
        
        # 交互式探索示例
        print("\nInteractive pixel exploration (random pixel):")
        visualizer.interactive_pixel_explorer()
        
        # 询问是否生成完整报告
        response = input("\nGenerate complete analysis report? (y/n): ")
        if response.lower() in ['y', 'yes']:
            visualizer.generate_report()
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")


if __name__ == "__main__":
    main()