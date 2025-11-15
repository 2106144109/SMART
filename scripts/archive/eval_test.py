#!/usr/bin/env python3
"""
测试集评估脚本
专门用于在测试集上评估训练好的模型
"""

from argparse import ArgumentParser
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.datasets.maritime_dataset import MaritimeDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder, MaritimeTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/train/train_maritime.yaml")
    parser.add_argument('--pretrain_ckpt', type=str, required=True, help="Path to checkpoint file")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help="Which split to evaluate on")
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 SMART Maritime 模型评估")
    print("="*80)
    print(f"📂 配置文件: {args.config}")
    print(f"💾 模型检查点: {args.pretrain_ckpt}")
    print(f"📊 评估数据集: {args.split}")
    print("="*80)
    
    config = load_config_act(args.config)
    data_config = config.Dataset
    
    # 根据数据集类型选择不同的Dataset和Transform
    dataset_classes = {
        "scalable": MultiDataset,
        "maritime": MaritimeDataset,
    }
    
    transform_classes = {
        "scalable": WaymoTargetBuilder,
        "maritime": MaritimeTargetBuilder,
    }
    
    dataset_class = dataset_classes[data_config.dataset]
    transform_class = transform_classes[data_config.dataset]
    
    # 根据split选择数据目录
    if args.split == 'test':
        raw_dir = data_config.test_raw_dir
        processed_dir = data_config.test_processed_dir
    else:
        raw_dir = data_config.val_raw_dir
        processed_dir = data_config.val_processed_dir
    
    print(f"\n📁 加载{args.split}数据集...")
    eval_dataset = dataset_class(
        root=data_config.root, 
        split=args.split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=transform_class(config.Model.num_historical_steps, config.Model.decoder.num_future_steps)
    )
    
    print(f"   数据集大小: {len(eval_dataset)} 个样本")
    
    # 创建DataLoader
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=data_config.val_batch_size if hasattr(data_config, 'val_batch_size') else data_config.batch_size,
        shuffle=False, 
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory, 
        persistent_workers=True if data_config.num_workers > 0 else False
    )
    
    print(f"\n🧠 加载模型...")
    logger = Logging().log(level='INFO')
    model = SMART(config.Model)
    model.inference_token = True  # 启用推理度量（minADE/minFDE）
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)
    
    # 创建Trainer并评估
    print(f"\n⚡ 开始评估...")
    trainer_config = config.Trainer
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy='ddp_find_unused_parameters_false', 
        num_sanity_val_steps=0
    )
    
    results = trainer.validate(model, dataloader)
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)
    
    if results:
        print("\n📊 评估结果:")
        for key, value in results[0].items():
            print(f"   {key}: {value:.4f}")
    
    print("\n" + "="*80)

