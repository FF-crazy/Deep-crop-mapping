#!/usr/bin/env python3
"""
Transformer模型训练脚本
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

from Transformer.model import create_transformer_model
from Transformer.dataset import prepare_data, save_data_info
from Transformer.utils import (
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
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.5, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss用于分割任务，特别适用于类别不平衡
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C, H, W), targets: (N, H, W)
        num_classes = inputs.size(1)
        
        # 将targets转换为one-hot编码
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 计算softmax概率
        inputs_soft = torch.softmax(inputs, dim=1)
        
        # 计算Dice系数
        dice_scores = []
        for i in range(num_classes):
            pred = inputs_soft[:, i]
            target = targets_one_hot[:, i]
            
            intersection = (pred * target).sum(dim=[1, 2])
            union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)  # (N, C)
        dice_loss = 1 - dice_scores.mean(dim=1)  # (N,)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数：Focal Loss + Dice Loss + 交叉熵
    """
    def __init__(
        self, 
        alpha: torch.Tensor = None, 
        gamma: float = 2.5,
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        ce_weight: float = 0.2,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # 确保权重和为1
        total_weight = focal_weight + dice_weight + ce_weight
        self.focal_weight /= total_weight
        self.dice_weight /= total_weight
        self.ce_weight /= total_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        
        combined = (self.focal_weight * focal + 
                   self.dice_weight * dice + 
                   self.ce_weight * ce)
        
        return combined


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """
    训练一个epoch - Transformer版本
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # 混合精度训练
        with autocast():
            outputs = model(data)
            # outputs shape: (batch, height, width, num_classes)
            # targets shape: (batch, height, width)
            
            # 重塑以计算损失
            outputs = outputs.permute(0, 3, 1, 2)  # (batch, num_classes, height, width)
            loss = criterion(outputs, targets)
            
            # 梯度累积：将损失除以累积步数
            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积逻辑
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # 如果使用per-batch调度器
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()
        
        # 统计
        total_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.numel()
        correct += predicted.eq(targets).sum().item()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新进度条
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{current_lr:.2e}'
        })
    
    # 处理最后一批次的梯度累积
    if len(train_loader) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
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
    主训练函数 - Transformer版本
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
        num_workers=config['num_workers'],
        augment_train=config.get('augment_train', True)
    )
    
    # 保存数据信息
    save_data_info(data_info, f"{config['save_dir']}/data_info.pkl")
    
    # 创建模型
    model = create_transformer_model(
        input_channels=config['input_channels'],
        temporal_steps=config['temporal_steps'],
        num_classes=data_info['num_classes'],
        patch_size=config.get('model_patch_size', 8),
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 损失函数
    if config.get('use_combined_loss', False):
        criterion = CombinedLoss(
            alpha=data_info['class_weights'].to(device),
            gamma=config.get('focal_gamma', 2.5),
            focal_weight=config.get('focal_weight', 0.5),
            dice_weight=config.get('dice_weight', 0.3),
            ce_weight=config.get('ce_weight', 0.2),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        print(f"Using Combined Loss (Focal+Dice+CE with label_smoothing={config.get('label_smoothing', 0.1)})")
    elif config['use_focal_loss']:
        criterion = FocalLoss(
            alpha=data_info['class_weights'].to(device), 
            gamma=config.get('focal_gamma', 2.5)
        )
        print(f"Using Focal Loss (gamma={config.get('focal_gamma', 2.5)})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=data_info['class_weights'].to(device),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        print("Using Cross Entropy Loss with label smoothing")
    
    # 优化器 - 为Transformer调整参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler_type = config.get('scheduler_type', 'cosine')
    
    if scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', config['learning_rate'] * 10),
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=config.get('pct_start', 0.3),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 1e4),
            anneal_strategy=config.get('anneal_strategy', 'cos')
        )
        scheduler_step_per_batch = True
        print(f"Using OneCycleLR (max_lr={config.get('max_lr', config['learning_rate'] * 10):.2e})")
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('min_lr', config['learning_rate'] * 1e-3)
        )
        scheduler_step_per_batch = False
        print("Using CosineAnnealingWarmRestarts")
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=config.get('min_lr', config['learning_rate'] * 1e-4)
        )
        scheduler_step_per_batch = False
        print("Using ReduceLROnPlateau")
    else:
        scheduler = None
        scheduler_step_per_batch = False
        print("No learning rate scheduler")
    
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
        wandb.init(project='crop-mapping-transformer', config=config)
    
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
            model, train_loader, criterion, optimizer, device, scaler, epoch,
            scheduler=scheduler if scheduler_step_per_batch else None,
            scheduler_step_per_batch=scheduler_step_per_batch,
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0)
        )
        
        # 验证
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, data_info['num_classes']
        )
        
        # 学习率调度（对于非per-batch调度器）
        if scheduler is not None and not scheduler_step_per_batch:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
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
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'best_miou': best_miou,
                'config': config,
                'history': history
            }, f"{config['save_dir']}/checkpoint_epoch_{epoch}.pth")
    
    # 保存训练历史
    with open(f"{config['save_dir']}/history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练曲线
    plot_training_history(
        history, 
        save_path=f"{config['save_dir']}/training_curves.png",
        title="Transformer Training History"
    )
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Transformer model for crop mapping')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='./dataset', help='Path to dataset')
    parser.add_argument('--patch-size', type=int, default=64, help='Input patch size')
    parser.add_argument('--stride', type=int, default=32, help='Stride for patches')
    
    # 模型参数
    parser.add_argument('--model-patch-size', type=int, default=8, help='Model patch size for tokenization')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm for clipping')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler-type', type=str, default='cosine', 
                       choices=['onecycle', 'cosine', 'plateau', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--max-lr', type=float, default=None, help='Max learning rate for OneCycleLR')
    parser.add_argument('--min-lr', type=float, default=None, help='Min learning rate')
    
    # 损失函数参数
    parser.add_argument('--use-combined-loss', action='store_true', help='Use combined loss (Focal+Dice+CE)')
    parser.add_argument('--focal-gamma', type=float, default=2.5, help='Focal loss gamma parameter')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    
    # 其他参数
    parser.add_argument('--save-dir', type=str, default='./Transformer/checkpoints', 
                       help='Directory to save models')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    
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
        'model_patch_size': args.model_patch_size,
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,
        
        # 损失函数配置
        'use_focal_loss': True,
        'use_combined_loss': args.use_combined_loss,
        'focal_gamma': args.focal_gamma,
        'label_smoothing': args.label_smoothing,
        
        # 学习率调度配置
        'scheduler_type': args.scheduler_type,
        'max_lr': args.max_lr or args.learning_rate * 10,
        'min_lr': args.min_lr or args.learning_rate * 1e-4,
        'T_0': 10,
        'T_mult': 2,
        
        # 数据增强
        'augment_train': True,
        
        # 其他参数
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