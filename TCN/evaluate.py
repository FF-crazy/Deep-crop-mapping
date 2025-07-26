#!/usr/bin/env python3
"""
模型评估脚本
对训练好的TCN模型进行全面评估
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

from model import create_tcn_model
from dataset import prepare_data, load_data_info
from utils import (
    load_checkpoint,
    calculate_metrics,
    plot_confusion_matrix,
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
    评估模型性能
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Evaluating model on test set...")
    
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
    生成评估报告
    """
    report = []
    report.append("="*60)
    report.append("TCN MODEL EVALUATION REPORT")
    report.append("="*60)
    
    # 总体指标
    report.append("\n1. OVERALL METRICS")
    report.append("-"*30)
    report.append(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    report.append(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    report.append(f"Mean IoU: {metrics['mean_iou']:.4f}")
    
    # 每类指标
    report.append("\n2. PER-CLASS METRICS")
    report.append("-"*30)
    report.append(f"{'Class':<20} {'Accuracy':<10} {'IoU':<10} {'F1-Score':<10}")
    report.append("-"*50)
    
    for i, class_name in metrics['class_names'].items():
        acc = metrics['per_class_accuracy'][i]
        iou = metrics['per_class_iou'][i]
        f1 = metrics['classification_report'][str(i)]['f1-score'] if str(i) in metrics['classification_report'] else 0
        report.append(f"{class_name:<20} {acc:<10.4f} {iou:<10.4f} {f1:<10.4f}")
    
    # 保存报告
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Evaluation report saved to {save_path}")
    
    # 打印到控制台
    print('\n'.join(report))


def analyze_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Dict[int, str],
    save_dir: str
):
    """
    分析预测错误
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
    
    print("\nMost common misclassifications:")
    print(f"{'True Class':<20} {'Predicted Class':<20} {'Count':<10}")
    print("-"*50)
    
    for (true_class, pred_class), count in most_common_errors:
        true_name = class_names[true_class]
        pred_name = class_names[pred_class]
        print(f"{true_name:<20} {pred_name:<20} {count:<10}")
    
    # 保存错误分析
    error_df = pd.DataFrame(error_pairs, columns=['True', 'Predicted'])
    error_df['TrueClass'] = error_df['True'].map(class_names)
    error_df['PredictedClass'] = error_df['Predicted'].map(class_names)
    
    error_summary = error_df.groupby(['TrueClass', 'PredictedClass']).size().reset_index(name='Count')
    error_summary = error_summary.sort_values('Count', ascending=False)
    
    error_summary.to_csv(f"{save_dir}/error_analysis.csv", index=False)


def plot_class_performance(metrics: Dict[str, Any], save_path: str = None):
    """
    绘制每类性能图表
    """
    class_names = [metrics['class_names'][i] for i in range(len(metrics['per_class_accuracy']))]
    accuracies = metrics['per_class_accuracy']
    ious = metrics['per_class_iou']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 准确率条形图
    bars1 = ax1.bar(range(len(class_names)), accuracies, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Class Accuracy')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.axhline(y=metrics['mean_accuracy'], color='red', linestyle='--', label=f"Mean: {metrics['mean_accuracy']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # IoU条形图
    bars2 = ax2.bar(range(len(class_names)), ious, color='lightgreen', edgecolor='darkgreen')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('IoU')
    ax2.set_title('Per-Class Intersection over Union (IoU)')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.axhline(y=metrics['mean_iou'], color='red', linestyle='--', label=f"Mean: {metrics['mean_iou']:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_model_predictions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Dict[int, str],
    num_samples: int = 5,
    save_dir: str = None
):
    """
    可视化模型预测结果
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
        save_path=f"{save_dir}/predictions_visualization.png" if save_dir else None
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate TCN model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data-path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--save-dir', type=str, default='./TCN/evaluation', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    
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
        num_workers=4
    )
    
    # 创建并加载模型
    from compat import load_checkpoint
    checkpoint = load_checkpoint(args.model_path, map_location=str(device))
    
    # 从检查点获取模型配置
    config = checkpoint.get('config', {})
    
    model = create_tcn_model(
        input_channels=config.get('input_channels', 8),
        temporal_steps=config.get('temporal_steps', 28),
        num_classes=data_info['num_classes'],
        tcn_channels=config.get('tcn_channels', [64, 128, 256])
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    print(f"Best mIoU during training: {checkpoint.get('best_miou', 'N/A')}")
    
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
        save_path=f"{save_dir}/confusion_matrix.png"
    )
    
    # 绘制类别性能
    plot_class_performance(metrics, save_path=f"{save_dir}/class_performance.png")
    
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
    
    print(f"\nEvaluation complete! Results saved to {save_dir}")


if __name__ == "__main__":
    main()