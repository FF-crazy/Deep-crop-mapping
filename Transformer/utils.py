#!/usr/bin/env python3
"""
工具函数模块 - Transformer版本
包含模型保存/加载、早停、评估指标计算、可视化等功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, List, Tuple
import seaborn as sns
from pathlib import Path
import json


class EarlyStopping:
    """
    早停机制，防止过拟合
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0, save_path: str = 'checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        
    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """保存模型当验证损失下降时"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


def save_checkpoint(state: Dict[str, Any], filepath: str):
    """
    保存训练检查点
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    加载训练检查点
    """
    from .compat import load_checkpoint as compat_load_checkpoint
    checkpoint = compat_load_checkpoint(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def save_model(model: torch.nn.Module, save_path: str, model_info: Dict[str, Any] = None):
    """
    保存模型用于部署
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'model_type': 'transformer'
    }
    
    if model_info is not None:
        save_dict.update(model_info)
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")


def load_model(model_path: str, model: torch.nn.Module) -> Dict[str, Any]:
    """
    加载模型用于推理
    """
    from .compat import load_checkpoint as compat_load_checkpoint
    checkpoint = compat_load_checkpoint(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return checkpoint


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """
    计算各种评估指标
    """
    # 基本指标
    overall_accuracy = np.mean(predictions == targets)
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    # 每类指标
    per_class_accuracy = []
    per_class_iou = []
    per_class_precision = []
    per_class_recall = []
    
    for i in range(num_classes):
        # 类别准确率
        mask = targets == i
        if mask.sum() > 0:
            class_acc = np.mean(predictions[mask] == i)
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0)
        
        # IoU计算
        intersection = np.logical_and(targets == i, predictions == i).sum()
        union = np.logical_or(targets == i, predictions == i).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0
        per_class_iou.append(iou)
        
        # 精确率和召回率
        tp = intersection
        fp = (predictions == i).sum() - tp
        fn = (targets == i).sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        per_class_precision.append(precision)
        per_class_recall.append(recall)
    
    # 平均指标
    mean_accuracy = np.mean(per_class_accuracy)
    mean_iou = np.mean(per_class_iou)
    mean_precision = np.mean(per_class_precision)
    mean_recall = np.mean(per_class_recall)
    
    # F1分数等
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'mean_accuracy': mean_accuracy,
        'mean_iou': mean_iou,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'per_class_accuracy': per_class_accuracy,
        'per_class_iou': per_class_iou,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = None, title: str = "Transformer Confusion Matrix"):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: str = None, title: str = "Transformer Training History"):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 损失曲线
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    ax.plot(history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mIoU曲线
    ax = axes[1, 0]
    ax.plot(history['val_miou'], label='Val mIoU', color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Mean IoU History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax = axes[1, 1]
    ax.plot(history['learning_rates'], label='Learning Rate', color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_attention_maps(
    attention_weights: torch.Tensor,
    input_patches: torch.Tensor,
    save_path: str = None,
    num_heads: int = 4
):
    """
    可视化Transformer的注意力权重
    
    Args:
        attention_weights: 注意力权重 (batch, num_heads, num_patches, num_patches)
        input_patches: 输入patches (batch, num_patches, embed_dim)
        save_path: 保存路径
        num_heads: 显示的注意力头数量
    """
    batch_idx = 0  # 显示第一个样本
    
    # 选择前几个注意力头
    attention_to_plot = attention_weights[batch_idx, :num_heads].cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_heads, 4)):
        ax = axes[i]
        
        # 显示注意力图
        im = ax.imshow(attention_to_plot[i], cmap='viridis')
        ax.set_title(f'Attention Head {i+1}')
        ax.set_xlabel('Key Patches')
        ax.set_ylabel('Query Patches')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_performance(metrics: Dict[str, Any], save_path: str = None, title: str = "Transformer Per-Class Performance"):
    """
    绘制每类性能图表 - Transformer版本
    """
    if 'class_names' in metrics and 1 in metrics['class_names']:
        # 如果使用1-8标签，转换为0-7
        class_names = [metrics['class_names'][i+1] for i in range(len(metrics['per_class_accuracy']))]
    else:
        # 如果已经是0-7标签
        class_names = [metrics['class_names'][i] for i in range(len(metrics['per_class_accuracy']))]
    
    accuracies = metrics['per_class_accuracy']
    ious = metrics['per_class_iou']
    precisions = metrics['per_class_precision']
    recalls = metrics['per_class_recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # 准确率条形图
    ax = axes[0, 0]
    bars1 = ax.bar(range(len(class_names)), accuracies, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=metrics['mean_accuracy'], color='red', linestyle='--', 
               label=f"Mean: {metrics['mean_accuracy']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU条形图
    ax = axes[0, 1]
    bars2 = ax.bar(range(len(class_names)), ious, color='lightgreen', edgecolor='darkgreen')
    ax.set_xlabel('Classes')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=metrics['mean_iou'], color='red', linestyle='--', 
               label=f"Mean: {metrics['mean_iou']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 精确率条形图
    ax = axes[1, 0]
    bars3 = ax.bar(range(len(class_names)), precisions, color='lightcoral', edgecolor='darkred')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Precision')
    ax.set_title('Per-Class Precision')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=metrics['mean_precision'], color='blue', linestyle='--', 
               label=f"Mean: {metrics['mean_precision']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 召回率条形图
    ax = axes[1, 1]
    bars4 = ax.bar(range(len(class_names)), recalls, color='lightsalmon', edgecolor='darkred')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=metrics['mean_recall'], color='blue', linestyle='--', 
               label=f"Mean: {metrics['mean_recall']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    num_samples: int = 4,
    save_path: str = None,
    title: str = "Transformer Predictions"
):
    """
    可视化预测结果 - Transformer版本
    """
    # 选择要显示的样本
    indices = np.random.choice(images.shape[0], min(num_samples, images.shape[0]), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    fig.suptitle(title, fontsize=16)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, i in enumerate(indices):
        # 显示RGB图像（使用前3个波段的第一个时间步）
        img = images[i, :, :, 0, :3].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到0-1
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Input (RGB from first timepoint)')
        axes[idx, 0].axis('off')
        
        # 显示真实标签
        target_img = targets[i].cpu().numpy()
        im1 = axes[idx, 1].imshow(target_img, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # 显示预测结果
        pred_img = predictions[i].cpu().numpy()
        im2 = axes[idx, 2].imshow(pred_img, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # 计算该样本的准确率
        accuracy = np.mean(target_img == pred_img)
        axes[idx, 2].text(0.02, 0.98, f'Acc: {accuracy:.2%}', 
                         transform=axes[idx, 2].transAxes,
                         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                         verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def export_onnx(model: torch.nn.Module, save_path: str, input_shape: Tuple[int, ...]):
    """
    导出模型为ONNX格式
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape[1:])  # 批次大小为1
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to ONNX format: {save_path}")


if __name__ == "__main__":
    # 测试各个功能
    print("Transformer Utils module loaded successfully!")
    
    # 测试指标计算
    predictions = np.random.randint(0, 8, 1000)
    targets = np.random.randint(0, 8, 1000)
    
    metrics = calculate_metrics(predictions, targets, num_classes=8)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Precision: {metrics['mean_precision']:.4f}")
    print(f"Mean Recall: {metrics['mean_recall']:.4f}")