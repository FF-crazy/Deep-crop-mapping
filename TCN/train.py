#!/usr/bin/env python3
"""
TCN模型训练脚本
包含完整的训练流程、损失函数、优化器配置等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import wandb
import argparse
from tqdm import tqdm

from model import create_tcn_model
from dataset import prepare_data, save_data_info
from utils import (
    save_checkpoint, 
    load_checkpoint, 
    EarlyStopping,
    calculate_metrics,
    plot_training_history
)


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    epoch: int
) -> Tuple[float, float]:
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            outputs = model(data)
            # outputs shape: (batch, height, width, num_classes)
            # targets shape: (batch, height, width)
            
            # 重塑以计算损失
            outputs = outputs.permute(0, 3, 1, 2)  # (batch, num_classes, height, width)
            loss = criterion(outputs, targets)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.numel()
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Tuple[float, float, Dict[str, float]]:
    """
    验证模型
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            outputs = outputs.permute(0, 3, 1, 2)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 计算指标
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    val_loss = total_loss / len(val_loader)
    
    return val_loss, metrics['overall_accuracy'], metrics


def train_model(config: Dict[str, Any]):
    """
    主训练函数
    """
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 准备数据
    train_loader, val_loader, test_loader, data_info = prepare_data(
        data_path=config['data_path'],
        patch_size=config['patch_size'],
        stride=config['stride'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 保存数据信息
    save_data_info(data_info, f"{config['save_dir']}/data_info.pkl")
    
    # 创建模型
    model = create_tcn_model(
        input_channels=config['input_channels'],
        temporal_steps=config['temporal_steps'],
        num_classes=data_info['num_classes'],
        tcn_channels=config['tcn_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout']
    ).to(device)
    
    # 损失函数
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=data_info['class_weights'].to(device), gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=data_info['class_weights'].to(device))
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult']
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config['patience'],
        verbose=True,
        save_path=f"{config['save_dir']}/best_model.pth"
    )
    
    # 初始化wandb（可选）
    if config.get('use_wandb', False):
        wandb.init(project='crop-mapping-tcn', config=config)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_miou': [],
        'learning_rates': []
    }
    
    # 训练循环
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_miou = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # 验证
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, data_info['num_classes']
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_miou'].append(val_metrics['mean_iou'])
        history['learning_rates'].append(current_lr)
        
        # 打印结果
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Val mIoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # wandb记录
        if config.get('use_wandb', False):
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_miou': val_metrics['mean_iou'],
                'learning_rate': current_lr,
                'epoch': epoch
            })
        
        # 保存最佳模型
        if val_metrics['mean_iou'] > best_miou:
            best_miou = val_metrics['mean_iou']
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_miou': best_miou,
                'config': config,
                'metrics': val_metrics
            }, f"{config['save_dir']}/best_model.pth")
            print(f"  New best model saved! (mIoU: {best_miou:.4f})")
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        # 定期保存检查点
        if epoch % config['save_interval'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_miou': best_miou,
                'config': config,
                'history': history
            }, f"{config['save_dir']}/checkpoint_epoch_{epoch}.pth")
    
    # 保存训练历史
    with open(f"{config['save_dir']}/history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练曲线
    plot_training_history(history, save_path=f"{config['save_dir']}/training_curves.png")
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train TCN model for crop mapping')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    parser.add_argument('--stride', type=int, default=32, help='Stride for patches')
    
    # 模型参数
    parser.add_argument('--tcn-channels', type=int, nargs='+', default=[64, 128, 256], 
                       help='TCN channel sizes')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # 其他参数
    parser.add_argument('--save-dir', type=str, default='./TCN/checkpoints', 
                       help='Directory to save models')
    parser.add_argument('--use-wandb', default=False, help='Use Weights & Biases')
    
    args = parser.parse_args()
    
    # 配置字典
    config = {
        'data_path': args.data_path,
        'patch_size': args.patch_size,
        'stride': args.stride,
        'test_size': 0.2,
        'val_size': 0.1,
        'input_channels': 8,
        'temporal_steps': 28,
        'tcn_channels': args.tcn_channels,
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'use_focal_loss': True,
        'T_0': 10,
        'T_mult': 2,
        'patience': 15,
        'save_interval': 10,
        'num_workers': 4,
        'save_dir': args.save_dir,
        'use_wandb': args.use_wandb
    }
    
    # 创建保存目录
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(f"{config['save_dir']}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # 开始训练
    train_model(config)


if __name__ == "__main__":
    main()