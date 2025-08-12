#!/usr/bin/env python3
"""
模型评估脚本 - Transformer版本
对训练好的Transformer模型进行全面评估
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from typing import Dict, Any, Tuple
from tqdm import tqdm
import pandas as pd

from model import create_transformer_model
from dataset import prepare_data, load_data_info
from utils import (
    load_checkpoint,
    calculate_metrics,
    plot_confusion_matrix,
    plot_class_performance,
    visualize_predictions
)


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Dict[int, str]
) -> Dict[str, Any]:
    """
    评估模型性能 - Transformer版本
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Evaluating Transformer model on test set...")
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Testing'):
            data = data.to(device)
            targets_cpu = targets.numpy()
            
            outputs = model(data)  # (batch, height, width, num_classes)
            probs = torch.softmax(outputs, dim=-1)
            
            _, predicted = outputs.max(-1)
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets_cpu)
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # 计算指标
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    
    # 添加类别名称
    metrics['class_names'] = class_names
    
    return metrics, all_preds, all_targets, all_probs


def generate_evaluation_report(
    metrics: Dict[str, Any],
    save_path: str
):
    """
    生成评估报告 - Transformer版本
    """
    report = []
    report.append("="*60)
    report.append("TRANSFORMER MODEL EVALUATION REPORT")
    report.append("="*60)
    
    # 总体指标
    report.append("\n1. OVERALL METRICS")
    report.append("-"*30)
    report.append(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    report.append(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    report.append(f"Mean IoU: {metrics['mean_iou']:.4f}")
    report.append(f"Mean Precision: {metrics['mean_precision']:.4f}")
    report.append(f"Mean Recall: {metrics['mean_recall']:.4f}")
    
    # 每类指标
    report.append("\n2. PER-CLASS METRICS")
    report.append("-"*30)
    report.append(f"{'Class':<20} {'Accuracy':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    report.append("-"*80)
    
    class_names = metrics['class_names']
    for i in range(len(metrics['per_class_accuracy'])):
        class_name = class_names.get(i, f"Class_{i}")
        acc = metrics['per_class_accuracy'][i]
        iou = metrics['per_class_iou'][i]
        precision = metrics['per_class_precision'][i]
        recall = metrics['per_class_recall'][i]
        f1 = metrics['classification_report'][str(i)]['f1-score'] if str(i) in metrics['classification_report'] else 0
        report.append(f"{class_name:<20} {acc:<10.4f} {iou:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Transformer特定分析
    report.append("\n3. TRANSFORMER SPECIFIC ANALYSIS")
    report.append("-"*30)
    report.append("This is a Vision Transformer model specifically designed for")
    report.append("multi-spectral time series crop mapping. The model uses:")
    report.append("- Patch-based tokenization with temporal feature extraction")
    report.append("- Multi-head self-attention mechanisms") 
    report.append("- Positional encoding for spatial relationships")
    report.append("- Advanced data augmentation including temporal masking")
    
    # 保存报告
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Evaluation report saved to {save_path}")
    
    # 打印到控制台
    print('\n'.join(report))


def analyze_attention_patterns(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 3
):
    """
    分析Transformer的注意力模式
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(test_loader)
    batch_data, batch_targets = next(data_iter)
    batch_data = batch_data.to(device)
    
    # 选择样本进行分析
    sample_indices = np.random.choice(batch_data.shape[0], min(num_samples, batch_data.shape[0]), replace=False)
    
    print(f"Analyzing attention patterns for {len(sample_indices)} samples...")
    
    with torch.no_grad():
        # 这里需要修改模型以返回注意力权重
        # 由于当前模型实现没有返回注意力权重，我们创建一个简单的可视化
        
        for idx, sample_idx in enumerate(sample_indices):
            sample_data = batch_data[sample_idx:sample_idx+1]
            sample_target = batch_targets[sample_idx]
            
            # 获取预测
            outputs = model(sample_data)
            _, predicted = outputs.max(-1)
            
            # 创建简单的特征可视化
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # 原始RGB图像（第一个时间步的前3个波段）
            img_data = sample_data[0, :, :, 0, :3].cpu().numpy()
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            axes[0, 0].imshow(img_data)
            axes[0, 0].set_title('Input RGB (first timepoint)')
            axes[0, 0].axis('off')
            
            # 真实标签
            axes[0, 1].imshow(sample_target.numpy(), cmap='tab10')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # 预测结果
            pred_img = predicted[0].cpu().numpy()
            axes[1, 0].imshow(pred_img, cmap='tab10')
            axes[1, 0].set_title('Prediction')
            axes[1, 0].axis('off')
            
            # 预测置信度
            confidence = torch.softmax(outputs, dim=-1).max(-1)[0][0].cpu().numpy()
            im = axes[1, 1].imshow(confidence, cmap='viridis')
            axes[1, 1].set_title('Prediction Confidence')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/attention_analysis_sample_{idx+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Attention analysis saved to {save_dir}")


def analyze_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Dict[int, str],
    save_dir: str
):
    """
    分析预测错误 - Transformer版本
    """
    # 错误样本
    errors = predictions != targets
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    # 错误类型统计
    error_pairs = []
    for idx in error_indices:
        error_pairs.append((targets[idx], predictions[idx]))
    
    # 统计最常见的错误
    from collections import Counter
    error_counter = Counter(error_pairs)
    most_common_errors = error_counter.most_common(10)
    
    print("\nMost common misclassifications (Transformer):")
    print(f"{'True Class':<20} {'Predicted Class':<20} {'Count':<10}")
    print("-"*50)
    
    for (true_class, pred_class), count in most_common_errors:
        true_name = class_names.get(true_class, f"Class_{true_class}")
        pred_name = class_names.get(pred_class, f"Class_{pred_class}")
        print(f"{true_name:<20} {pred_name:<20} {count:<10}")
    
    # 保存错误分析
    error_df = pd.DataFrame(error_pairs, columns=['True', 'Predicted'])
    error_df['TrueClass'] = error_df['True'].map(lambda x: class_names.get(x, f"Class_{x}"))
    error_df['PredictedClass'] = error_df['Predicted'].map(lambda x: class_names.get(x, f"Class_{x}"))
    
    error_summary = error_df.groupby(['TrueClass', 'PredictedClass']).size().reset_index(name='Count')
    error_summary = error_summary.sort_values('Count', ascending=False)
    
    error_summary.to_csv(f"{save_dir}/error_analysis.csv", index=False)
    print(f"Error analysis saved to {save_dir}/error_analysis.csv")


def visualize_model_predictions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Dict[int, str],
    num_samples: int = 5,
    save_dir: str = None
):
    """
    可视化模型预测结果 - Transformer版本
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(test_loader)
    batch_data, batch_targets = next(data_iter)
    
    # 预测
    with torch.no_grad():
        batch_data_device = batch_data.to(device)
        outputs = model(batch_data_device)
        _, predictions = outputs.max(-1)
    
    # 可视化
    visualize_predictions(
        batch_data[:num_samples],
        batch_targets[:num_samples],
        predictions[:num_samples].cpu(),
        list(class_names.values()),
        num_samples=num_samples,
        save_path=f"{save_dir}/transformer_predictions_visualization.png" if save_dir else None,
        title="Transformer Predictions"
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data-path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--save-dir', type=str, default='./Transformer/evaluation', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--analyze-attention', action='store_true', help='Analyze attention patterns')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据信息
    model_dir = Path(args.model_path).parent
    data_info = load_data_info(f"{model_dir}/data_info.pkl")
    
    # 准备数据
    _, _, test_loader, _ = prepare_data(
        data_path=args.data_path,
        patch_size=data_info['patch_size'],
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=False
    )
    
    # 创建并加载模型
    from compat import load_checkpoint
    checkpoint = load_checkpoint(args.model_path, map_location=str(device))
    
    # 从检查点获取模型配置
    config = checkpoint.get('config', {})
    
    model = create_transformer_model(
        input_channels=config.get('input_channels', 8),
        temporal_steps=config.get('temporal_steps', 28),
        num_classes=data_info['num_classes'],
        patch_size=config.get('model_patch_size', 8),
        embed_dim=config.get('embed_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Transformer model loaded from {args.model_path}")
    print(f"Best mIoU during training: {checkpoint.get('best_miou', 'N/A')}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 评估模型
    metrics, predictions, targets, probs = evaluate_model(
        model, test_loader, device, 
        data_info['num_classes'],
        data_info['class_names']
    )
    
    # 生成报告
    generate_evaluation_report(metrics, f"{save_dir}/evaluation_report.txt")
    
    # 保存详细指标
    with open(f"{save_dir}/metrics.json", 'w') as f:
        # 处理numpy数组
        metrics_to_save = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_to_save, f, indent=4)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        list(data_info['class_names'].values()),
        save_path=f"{save_dir}/confusion_matrix.png",
        title="Transformer Confusion Matrix"
    )
    
    # 绘制类别性能
    plot_class_performance(
        metrics, 
        save_path=f"{save_dir}/class_performance.png",
        title="Transformer Per-Class Performance"
    )
    
    # 错误分析
    analyze_errors(predictions, targets, data_info['class_names'], save_dir)
    
    # 可视化预测
    if args.visualize:
        visualize_model_predictions(
            model, test_loader, device,
            data_info['class_names'],
            num_samples=5,
            save_dir=save_dir
        )
    
    # 注意力分析
    if args.analyze_attention:
        analyze_attention_patterns(
            model, test_loader, device,
            save_dir, num_samples=3
        )
    
    print(f"\nTransformer evaluation complete! Results saved to {save_dir}")


if __name__ == "__main__":
    main()