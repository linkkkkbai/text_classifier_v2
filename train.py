import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from models.multi_head import ResNetCropMultiAttr
from datasets.coco_loader import build_dataloader

from losses.focal_loss import MultiAttributeLoss
from utils.train_utils import load_config,train_iterations
import json


def main(args):

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

    seed = 42  # 或在配置文件中设置 cfg.training.seed
    # 设置Python随机模块
    import random
    random.seed(seed)
    
    # 设置NumPy随机种子
    import numpy as np
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 加载配置
    cfg = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    from torchvision import transforms as T

    train_loader = build_dataloader(cfg, cfg.data.train_json, batch_size=cfg.data.batch_size, is_train=True,num_workers=cfg.data.num_workers,shuffle=True)
    val_loader = build_dataloader(cfg, cfg.data.val_json, batch_size=cfg.data.batch_size, is_train=False, num_workers=cfg.data.num_workers,shuffle=False)

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # 构建模型
    model = ResNetCropMultiAttr(cfg)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = MultiAttributeLoss(cfg)


    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.training.lr,
    #     weight_decay=cfg.training.weight_decay
    # )
    optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 5e-5},      # 主干较小学习率
    {'params': model.type_head.parameters(), 'lr': 1e-4},     # 分类头较大学习率
], weight_decay=cfg.training.weight_decay)
    
    #学习率调度器
    if hasattr(cfg.training, "lr_scheduler") and cfg.training.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.training.total_iters,
            eta_min=cfg.training.lr * 0.01
        )
    else:
        scheduler = None
 
    
    # TensorBoard记录器
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 训练（以iteration为单位）
    metrics_history = train_iterations(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        total_iters=cfg.training.total_iters,
        validate_interval=cfg.training.validate_interval,
        checkpoint_interval=cfg.training.checkpoint_interval,
        output_dir=args.output_dir,
        writer=writer,
        scheduler=scheduler
    )
    
    # 保存训练指标历史
    with open(os.path.join(args.output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-attribute text classifier")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    args = parser.parse_args()
    
    main(args)