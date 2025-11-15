#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch

from smart.utils.config import load_config_act
from smart.utils.log import Logging
from smart.model import SMART
from smart.datasets.scalable_dataset import MultiDataset
from smart.datasets.maritime_dataset import MaritimeDataset
from smart.transforms import WaymoTargetBuilder, MaritimeTargetBuilder

# 复用现有的绘图/采样/指标函数，避免重复实现
import visualize_predictions_folium as vpf


def _parse_indices(indices_str: str, ds_len: int):
    """
    解析逗号分隔的索引字符串，过滤非法/越界，并去重保序
    """
    if not indices_str:
        return []
    raw = [s.strip() for s in indices_str.split(',') if s.strip()]
    parsed = []
    seen = set()
    for s in raw:
        if not s.isdigit():
            print(f"❌ 非法索引: {s}（已跳过）"); continue
        idx = int(s)
        if not (0 <= idx < ds_len):
            print(f"❌ 索引越界: {idx}（0 <= idx < {ds_len}，已跳过）"); continue
        if idx not in seen:
            seen.add(idx)
            parsed.append(idx)
    return parsed


def build_dataset(config, split: str):
    data_cfg = config.Dataset
    dataset_classes = {"scalable": MultiDataset, "maritime": MaritimeDataset}
    transform_classes = {"scalable": WaymoTargetBuilder, "maritime": MaritimeTargetBuilder}
    ds_name = data_cfg.dataset
    dataset_class = dataset_classes[ds_name]
    transform_class = transform_classes[ds_name]

    if split == 'test':
        raw_dir = data_cfg.test_raw_dir
        processed_dir = data_cfg.test_processed_dir
    else:
        raw_dir = data_cfg.val_raw_dir
        processed_dir = data_cfg.val_processed_dir

    ds = dataset_class(
        root=data_cfg.root,
        split=split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=transform_class(config.Model.num_historical_steps,
                                  config.Model.decoder.num_future_steps)
    )
    return ds


def main():
    parser = argparse.ArgumentParser(description="可视化若干指定/抽样场景：历史+GT未来+预测（含指标）")
    parser.add_argument('--config', type=str, default='configs/train/train_maritime.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])

    # 精确点名优先生效；否则按抽样策略
    parser.add_argument('--indices', type=str, default='', help='逗号分隔索引，如 "12,345,678"')
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--sample_mode', type=str, default=os.getenv('FOLIUM_SAMPLE_MODE', 'bucket'),
                        choices=['bucket', 'uniform', 'random'])
    parser.add_argument('--bucket_pick', type=str, default=os.getenv('FOLIUM_BUCKET_PICK', 'median'),
                        choices=['median', 'random', 'first', 'last'])
    parser.add_argument('--norm_stats', type=str, default=os.getenv('FOLIUM_NORM_STATS', ''))
    parser.add_argument('--output_dir', type=str, default='folium_pred_maps_selected')
    parser.add_argument('--no_save_map', action='store_true')
    args = parser.parse_args()

    print("="*80)
    print("🗺️  选择若干场景对比：历史 + 未来GT + 预测（含指标）")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"权重: {args.pretrain_ckpt}")
    print(f"数据: {args.split}")
    print(f"输出: {args.output_dir}")
    print("="*80)

    # 加载配置/数据
    config = load_config_act(args.config)
    ds = build_dataset(config, args.split)
    print(f"📂 数据量: {len(ds)}")

    # 准备模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = Logging().log(level='INFO')
    model = SMART(config.Model).to(device).eval()
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)

    # 归一化统计（可选）
    norm_stats = vpf._load_norm_stats(args.norm_stats)

    # 确定场景索引
    indices = _parse_indices(args.indices, len(ds))
    if not indices:
        if args.sample_mode == 'bucket':
            indices = vpf._pick_indices_bucket(ds, num_scenes=args.num_scenes, pick_mode=args.bucket_pick)
        elif args.sample_mode == 'uniform':
            indices = vpf._pick_indices_uniform(ds, num_scenes=args.num_scenes)
        else:
            seed_env = os.getenv('FOLIUM_SAMPLE_SEED')
            seed = int(seed_env) if (seed_env is not None and seed_env.strip() != '') else 0
            indices = vpf._pick_indices_random(ds, num_scenes=args.num_scenes, seed=seed)
    if not indices:
        print("❌ 无可视化样本。"); return

    os.makedirs(args.output_dir, exist_ok=True)

    # 默认中心（备用），最终会在绘制函数里按 scene_info/环境变量使用参考锚点
    center_lat, center_lon = 30.0, 120.0

    print(f"\n🗺️  开始可视化(共 {len(indices)} 个场景) ...")
    for out_idx, ds_idx in enumerate(indices):
        sample = ds[ds_idx].to(device)
        with torch.no_grad():
            pred = model.inference(sample)

        save_path = os.path.join(args.output_dir, f'scene_{out_idx:03d}.html')
        vpf._draw_scene_prediction_map(
            data=sample,
            pred=pred,
            output_path=save_path,
            scene_id=ds_idx,             # 报告原始数据集索引，便于复现
            center_lat=center_lat,
            center_lon=center_lon,
            norm_stats=norm_stats,
            save_map=not args.no_save_map
        )

    if not args.no_save_map:
        vpf._create_index_page(args.output_dir, len(indices))
    print("\n✅ 全部完成！")


if __name__ == '__main__':
    main()