#!/usr/bin/env python3
"""
测试Transformer模型实现的脚本
"""

def test_transformer_import():
    """测试模块导入"""
    try:
        # 测试所有必要的导入
        import torch
        print("✓ PyTorch导入成功")
        
        import torch.nn as nn
        print("✓ torch.nn导入成功")
        
        from Transformer.model import create_transformer_model, CropMappingTransformer
        print("✓ Transformer模型导入成功")
        
        from Transformer.dataset import prepare_data, CropMappingDataset
        print("✓ Transformer数据集导入成功")
        
        from Transformer.utils import calculate_metrics, plot_training_history
        print("✓ Transformer工具函数导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    try:
        import torch
        from Transformer.model import create_transformer_model
        
        print("\n开始测试Transformer模型创建...")
        
        # 创建模型
        model = create_transformer_model(
            input_channels=8,
            temporal_steps=28,
            num_classes=8,
            patch_size=8,
            embed_dim=256,
            num_layers=6,
            num_heads=8
        )
        
        print("✓ 模型创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 总参数量: {total_params:,}")
        print(f"✓ 可训练参数: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None


def test_forward_pass(model):
    """测试前向传播"""
    try:
        import torch
        
        print("\n开始测试前向传播...")
        
        # 创建测试输入
        batch_size = 2
        height, width = 64, 64
        temporal_steps = 28
        input_channels = 8
        
        dummy_input = torch.randn(batch_size, height, width, temporal_steps, input_channels)
        print(f"✓ 测试输入形状: {dummy_input.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (batch_size, height, width, 8)  # 8个类别
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 期望形状: {expected_shape}")
        
        if output.shape == expected_shape:
            print("✓ 前向传播测试成功!")
            return True
        else:
            print("✗ 输出形状不匹配")
            return False
            
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    try:
        import torch
        import torch.nn as nn
        from Transformer.train import FocalLoss, DiceLoss, CombinedLoss
        
        print("\n开始测试损失函数...")
        
        # 创建测试数据
        batch_size = 2
        num_classes = 8
        height, width = 32, 32
        
        # 模拟输出 (batch, num_classes, height, width)
        outputs = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # 测试Focal Loss
        focal_loss = FocalLoss(gamma=2.5)
        focal_result = focal_loss(outputs, targets)
        print(f"✓ Focal Loss: {focal_result.item():.4f}")
        
        # 测试Dice Loss
        dice_loss = DiceLoss()
        dice_result = dice_loss(outputs, targets)
        print(f"✓ Dice Loss: {dice_result.item():.4f}")
        
        # 测试组合损失
        combined_loss = CombinedLoss()
        combined_result = combined_loss(outputs, targets)
        print(f"✓ Combined Loss: {combined_result.item():.4f}")
        
        print("✓ 损失函数测试成功!")
        return True
        
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("TRANSFORMER模型实现测试")
    print("=" * 60)
    
    # 测试1: 模块导入
    print("\n1. 测试模块导入")
    if not test_transformer_import():
        print("模块导入失败，停止测试")
        return
    
    # 测试2: 模型创建
    print("\n2. 测试模型创建")
    model = test_model_creation()
    if model is None:
        print("模型创建失败，停止测试")
        return
    
    # 测试3: 前向传播
    print("\n3. 测试前向传播")
    if not test_forward_pass(model):
        print("前向传播测试失败")
        return
    
    # 测试4: 损失函数
    print("\n4. 测试损失函数")
    if not test_loss_functions():
        print("损失函数测试失败")
        return
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过! Transformer实现正常工作!")
    print("=" * 60)
    
    # 输出使用建议
    print("\n📋 使用建议:")
    print("1. 训练模型:")
    print("   python -m Transformer.train --data-path ./dataset --epochs 50")
    
    print("\n2. 评估模型:")
    print("   python -m Transformer.evaluate --model-path ./Transformer/checkpoints/best_model.pth")
    
    print("\n3. 推理预测:")
    print("   python -m Transformer.inference --model-path ./Transformer/checkpoints/best_model.pth --input-path ./dataset/x.npy")


if __name__ == "__main__":
    main()