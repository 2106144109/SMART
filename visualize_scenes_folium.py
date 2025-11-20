#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
import numpy as np
import torch
import folium

from torch_geometric.loader import DataLoader
from smart.utils.config import load_config_act
from smart.datasets.scalable_dataset import MultiDataset
from smart.datasets.maritime_dataset import MaritimeDataset
from smart.transforms import WaymoTargetBuilder, MaritimeTargetBuilder

# 复用既有可视化脚本中的工具函数（坐标反归一化 / 米->经纬度 / 轴交换翻转 / 开关解析）
try:
    from visualize_folium import (
        meters_to_lat_lon,
        denormalize_coordinates,
        _apply_transform_xy,
        _parse_axis_override,
    )
except Exception:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from visualize_folium import (
        meters_to_lat_lon,
        denormalize_coordinates,
        _apply_transform_xy,
        _parse_axis_override,
    )


def _load_norm_stats(path_or_none: str):
    if not path_or_none:
        return None
    try:
        with open(path_or_none, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        print("⚠️  norm_stats 加载失败，将以未归一化处理。")
        return None


def _get_anchor_from_sample(data, fallback_center):
    use_ref_anchor = os.getenv('FOLIUM_USE_REF_ANCHOR', '1') == '1'
    anchor_lat, anchor_lon, anchor_theta = fallback_center[0], fallback_center[1], 0.0
    if use_ref_anchor:
        scene_info = getattr(data, 'scene_info', None)
        if isinstance(scene_info, dict):
            anchor_lat = float(scene_info.get('ref_lat', anchor_lat))
            anchor_lon = float(scene_info.get('ref_lon', anchor_lon))
            anchor_theta = float(scene_info.get('ref_theta', anchor_theta))
            print(f"  [INFO] 使用窗口参考锚点: ({anchor_lat:.6f}, {anchor_lon:.6f}), theta={math.degrees(anchor_theta):.1f}°")
        else:
            meta = data.metadata if (hasattr(data, 'metadata') and isinstance(data.metadata, dict)) else {}
            anchor_lat = float(meta.get('origin_lat', anchor_lat))
            anchor_lon = float(meta.get('origin_lon', anchor_lon))
            print(f"  [INFO] 回退到场景原点为锚点: ({anchor_lat:.6f}, {anchor_lon:.6f})（无 ref_theta 不做旋回）")
    return use_ref_anchor, anchor_lat, anchor_lon, anchor_theta


def _pick_indices_bucket(ds, num_scenes: int, pick_mode: str = 'median'):
    file_to_ds_indices = {}
    for ds_idx, (file_idx, sample_idx) in enumerate(ds._sample_indices):
        file_to_ds_indices.setdefault(file_idx, []).append(ds_idx)

    unique_file_indices = sorted(file_to_ds_indices.keys())
    num_scenes = min(num_scenes, len(unique_file_indices))
    if num_scenes <= 0:
        return []

    sel = np.linspace(0, len(unique_file_indices) - 1, num=num_scenes, dtype=int)
    chosen_files = [unique_file_indices[i] for i in sel]

    def _pick(bucket):
        if not bucket:
            return None
        if pick_mode == 'random':
            return int(bucket[np.random.randint(0, len(bucket))])
        if pick_mode == 'first':
            return int(bucket[0])
        if pick_mode == 'last':
            return int(bucket[-1])
        return int(bucket[len(bucket) // 2])

    indices = []
    for fidx in chosen_files:
        bucket = file_to_ds_indices.get(fidx, [])
        ds_idx = _pick(bucket)
        if ds_idx is None:
            continue
        src_file = os.path.basename(ds._file_paths[fidx]) if hasattr(ds, '_file_paths') else str(fidx)
        sample_idx = ds._sample_indices[ds_idx][1]
        print(f"   [DEBUG] 分桶抽样: file_idx={fidx}, src={src_file}, bucket_size={len(bucket)}, pick={pick_mode}, ds_idx={ds_idx}, sample_idx={sample_idx}")
        indices.append(ds_idx)
    return indices


def _pick_indices_uniform(ds, num_scenes: int):
    total = len(ds)
    if total <= 0:
        return []
    num = int(min(num_scenes, total))
    return np.linspace(0, total - 1, num=num, dtype=int).tolist()


def _pick_indices_random(ds, num_scenes: int, seed: int = 0):
    total = len(ds)
    if total <= 0:
        return []
    num = int(min(num_scenes, total))
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=num, replace=False).tolist()


def _transform_point(xm, ym, use_ref_anchor, anchor_theta, swap_xy, flip_x, flip_y, anchor_lat, anchor_lon, norm_stats):
    if norm_stats is not None:
        xm, ym = denormalize_coordinates(xm, ym, norm_stats)
    xm, ym = _apply_transform_xy(xm, ym, swap_xy, flip_x, flip_y)
    if use_ref_anchor and anchor_theta != 0.0:
        ct, st = math.cos(anchor_theta), math.sin(anchor_theta)
        dx_world =  ct * xm - st * ym
        dy_world =  st * xm + ct * ym
    else:
        dx_world, dy_world = xm, ym
    lat, lon = meters_to_lat_lon(dx_world, dy_world, anchor_lat, anchor_lon)
    return lat, lon


def _draw_scene_gt_map(
    data, output_path, scene_id, center_lat, center_lon, norm_stats, save_map: bool = True
):
    # 地图中心尽量取锚点
    use_ref_anchor, anchor_lat, anchor_lon, anchor_theta = _get_anchor_from_sample(
        data, fallback_center=(center_lat, center_lon)
    )
    m = folium.Map(location=[anchor_lat, anchor_lon], zoom_start=15, tiles='OpenStreetMap', control_scale=True)

    # 轴策略：use_ref_anchor 时默认禁用自动推断（0,0,0），可用 FOLIUM_FORCE_TRANSFORM 强制覆盖
    forced, fswap, fflipx, fflipy = _parse_axis_override()
    _disable_default = '1' if use_ref_anchor else '0'
    disable_auto_axis = os.getenv('FOLIUM_DISABLE_AUTO_AXIS', _disable_default) == '1'
    if forced:
        swap_xy, flip_x, flip_y = fswap, fflipx, fflipy
        print(f"    [DEBUG] 轴变换(强制): swap_xy={swap_xy}, flip_x={flip_x}, flip_y={flip_y}")
    elif disable_auto_axis:
        swap_xy, flip_x, flip_y = False, False, False
        print(f"    [DEBUG] 轴变换(禁用自动): swap_xy={swap_xy}, flip_x={flip_x}, flip_y={flip_y}")
    else:
        swap_xy, flip_x, flip_y = False, False, False
        print(f"    [DEBUG] 轴变换(默认): swap_xy={swap_xy}, flip_x={flip_x}, flip_y={flip_y}")

    def _draw_map_segments(map_save, pt_token):
        if map_save is None or 'traj_pos' not in map_save:
            return

        traj_pos = map_save['traj_pos']
        if hasattr(traj_pos, 'cpu'):
            traj_pos = traj_pos.cpu().numpy()
        else:
            traj_pos = np.asarray(traj_pos)

        pl_type = None
        if pt_token is not None:
            pl_type = pt_token.get('pl_type')
            if hasattr(pl_type, 'cpu'):
                pl_type = pl_type.cpu().numpy()
            elif pl_type is not None:
                pl_type = np.asarray(pl_type)

        palette = [
            "#7f8c8d", "#95a5a6", "#16a085", "#27ae60", "#2980b9",
            "#8e44ad", "#2c3e50", "#f39c12", "#d35400", "#c0392b",
        ]

        for idx in range(traj_pos.shape[0]):
            coords = []
            for x, y in traj_pos[idx]:
                lat, lon = _transform_point(
                    x, y, use_ref_anchor, anchor_theta, swap_xy, flip_x, flip_y, anchor_lat, anchor_lon, norm_stats
                )
                coords.append([lat, lon])
            color = palette[idx % len(palette)]
            if pl_type is not None and idx < len(pl_type):
                color = palette[int(pl_type[idx]) % len(palette)]
            folium.PolyLine(coords, color=color, weight=1.2, opacity=0.55).add_to(m)

    # 地图矢量（若存在）
    map_save = data.get('map_save') if isinstance(data, dict) else getattr(data, 'map_save', None)
    pt_token = data.get('pt_token') if isinstance(data, dict) else getattr(data, 'pt_token', None)
    _draw_map_segments(map_save, pt_token)

    # 提取数据
    # 期望 agent.x: [N, T, 8]，其中 [:,:,0:2] = (x,y), [:,:,6] = theta
    if 'x' in data['agent']:
        feat = data['agent']['x'].cpu().numpy()
    else:
        # 兼容 position+heading 的结构
        pos = data['agent']['position'][:, :, :2].cpu().numpy()
        head = data['agent']['heading'].cpu().numpy()
        zeros = np.zeros((pos.shape[0], pos.shape[1], 8), dtype=pos.dtype)
        zeros[:, :, :2] = pos
        zeros[:, :, 6] = head
        feat = zeros

    pos   = feat[:, :, :2]
    heads = feat[:, :, 6]
    if 'valid_mask' in data['agent']:
        valid_mask = data['agent']['valid_mask'].cpu().numpy()
    else:
        valid_mask = np.ones(pos.shape[:2], dtype=bool)

    # 默认历史/未来步（若未知则平分）
    num_steps = pos.shape[1]
    num_his = int(os.getenv('FOLIUM_HIS_STEPS', '5'))
    num_fut = num_steps - num_his
    num_fut = max(0, num_fut)

    colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
        "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#8da0cb", "#e78ac3", "#1b9e77", "#d95f02"
    ]

    all_coords = []

    for agent_id in range(pos.shape[0]):
        color = colors[agent_id % len(colors)]

        # 历史
        hist_positions = pos[agent_id, :num_his, :]
        hist_valid = valid_mask[agent_id, :num_his]
        hist_coords = []
        for ok, (x, y) in zip(hist_valid.tolist(), hist_positions):
            if not ok:
                continue
            lat, lon = _transform_point(x, y, use_ref_anchor, anchor_theta, swap_xy, flip_x, flip_y, anchor_lat, anchor_lon, norm_stats)
            all_coords.append([lat, lon])
            folium.CircleMarker(location=[lat, lon], radius=2.5, color=color, weight=1, opacity=0.9,
                                fill=True, fill_color=color, fill_opacity=0.7).add_to(m)
            hist_coords.append([lat, lon])
        if len(hist_coords) > 1:
            folium.PolyLine(hist_coords, color=color, weight=2.0, opacity=0.8).add_to(m)

        hist_last_coord = hist_coords[-1] if len(hist_coords) > 0 else None

        # 未来 GT（绿色）
        fut_positions = pos[agent_id, num_his:num_his+num_fut, :]
        fut_valid = valid_mask[agent_id, num_his:num_his+num_fut]
        fut_coords = []
        for ok, (x, y) in zip(fut_valid.tolist(), fut_positions):
            if not ok:
                continue
            lat, lon = _transform_point(x, y, use_ref_anchor, anchor_theta, swap_xy, flip_x, flip_y, anchor_lat, anchor_lon, norm_stats)
            all_coords.append([lat, lon])
            fut_coords.append([lat, lon])
        if hist_last_coord is not None and len(fut_coords) > 0:
            fut_coords = [hist_last_coord] + fut_coords
        if len(fut_coords) > 1:
            folium.PolyLine(
                fut_coords,
                color="#2ecc71",
                weight=3.0,
                opacity=0.9,
                dash_array="6,4"
            ).add_to(m)

    # 视野自适应
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    if save_map:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        m.save(output_path)
        print(f"  ✅ 地图保存: {output_path}")


def _create_index_page(output_dir, num_scenes):
    html = ['''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMART Maritime - Scene Visualization</title>
<style>
body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}
.container{max-width:1200px;margin:0 auto;background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);} 
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;margin-top:30px;} 
.card{border:2px solid #3498db;border-radius:8px;padding:15px;text-align:center;transition:transform .2s;background:#f8fbff;} 
.card:hover{transform:translateY(-4px);box-shadow:0 5px 15px rgba(0,0,0,0.2);} 
.card a{text-decoration:none;color:#2c3e50;font-size:18px;font-weight:bold;} 
.legend{margin:10px 0;color:#444}
</style></head><body><div class="container">
<h1>🗺️ SMART Maritime - Scene Visualization</h1>
<div class="legend">图例：历史=原色实线；GT未来=绿色虚线</div>
<div class="grid">''']
    for i in range(num_scenes):
        html.append(f'''\n<div class="card"><a href="scene_{i:03d}.html" target="_blank">🌊 Scene {i}<br><small>Click to open</small></a></div>''')
    html.append('''</div></div></body></html>''')
    path = os.path.join(output_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(html))
    print(f"📑 索引已创建: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_maritime.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='folium_maps')
    parser.add_argument('--bucket_pick', type=str, default=os.getenv('FOLIUM_BUCKET_PICK', 'median'),
                        choices=['median', 'random', 'first', 'last'])
    parser.add_argument('--norm_stats', type=str, default=os.getenv('FOLIUM_NORM_STATS', ''))
    parser.add_argument('--sample_mode', type=str, default=os.getenv('FOLIUM_SAMPLE_MODE', 'bucket'),
                        choices=['bucket', 'uniform', 'random'])
    parser.add_argument('--no_save_map', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置与数据集
    config = load_config_act(args.config)
    data_cfg = config.Dataset

    dataset_classes = {
        "scalable": MultiDataset,
        "maritime": MaritimeDataset,
    }
    transform_classes = {
        "scalable": WaymoTargetBuilder,
        "maritime": MaritimeTargetBuilder,
    }
    dataset_class = dataset_classes[data_cfg.dataset]
    transform_class = transform_classes[data_cfg.dataset]

    if args.split == 'test':
        raw_dir = data_cfg.test_raw_dir
        processed_dir = data_cfg.test_processed_dir
    else:
        raw_dir = data_cfg.val_raw_dir
        processed_dir = data_cfg.val_processed_dir

    print(f"\n📁 加载数据集({args.split}) ...")
    ds = dataset_class(
        root=data_cfg.root,
        split=args.split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=transform_class(config.Model.num_historical_steps, config.Model.decoder.num_future_steps)
    )
    print(f"   数据量: {len(ds)}")

    # 归一化统计（可选）
    norm_stats = _load_norm_stats(args.norm_stats)

    # 抽样索引
    if args.sample_mode == 'bucket':
        indices = _pick_indices_bucket(ds, num_scenes=args.num_scenes, pick_mode=args.bucket_pick)
    elif args.sample_mode == 'uniform':
        indices = _pick_indices_uniform(ds, num_scenes=args.num_scenes)
    else:  # random
        seed_env = os.getenv('FOLIUM_SAMPLE_SEED')
        seed = int(seed_env) if (seed_env is not None and seed_env.strip() != '') else 0
        indices = _pick_indices_random(ds, num_scenes=args.num_scenes, seed=seed)
    if not indices:
        print("❌ 无可视化样本。"); return

    # 默认中心（备用）
    center_lat, center_lon = 30.0, 120.0

    print(f"\n🗺️  开始可视化场景(共 {len(indices)} 个) ...")
    for out_idx, ds_idx in enumerate(indices):
        sample = ds[ds_idx]
        save_path = os.path.join(args.output_dir, f'scene_{out_idx:03d}.html')
        _draw_scene_gt_map(
            data=sample,
            output_path=save_path,
            scene_id=out_idx,
            center_lat=center_lat,
            center_lon=center_lon,
            norm_stats=norm_stats,
            save_map=not args.no_save_map
        )

    if not args.no_save_map:
        _create_index_page(args.output_dir, len(indices))
    print("\n✅ 全部完成！")


if __name__ == '__main__':
    main()


