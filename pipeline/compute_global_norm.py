#!/usr/bin/env python3
"""
遍历所有处理好的海上场景数据（per_file/*.pkl），
对局部特征的关键维度计算全局均值与标准差，并保存为JSON。

使用方法：
  python scripts/compute_global_norm.py \
    --data_root /home/mahexing/ais_data_process/scene_generation/DI-MTP/data/per_file \
    --out /home/mahexing/SMART-main/data/maritime_global_norm.json
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保可以从仓库根目录导入模块
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maritime_scene_preprocessor import MaritimeScenePreprocessor


def list_pkl_files(data_root: str):
    files = [f for f in os.listdir(data_root) if f.endswith('.pkl')]
    files.sort()
    return files


def init_aggregator(keys):
    agg = {}
    for k in keys:
        agg[k] = {
            'sum': 0.0,
            'sumsq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'count': 0
        }
    return agg


def update_agg(agg, values):
    """values: dict[key] -> 1D numpy array"""
    for k, arr in values.items():
        if arr.size == 0:
            continue
        arr = arr.astype(np.float64, copy=False)
        agg[k]['sum'] += float(arr.sum())
        agg[k]['sumsq'] += float((arr * arr).sum())
        agg[k]['min'] = float(min(agg[k]['min'], float(arr.min())))
        agg[k]['max'] = float(max(agg[k]['max'], float(arr.max())))
        agg[k]['count'] += int(arr.size)


def merge_aggs(a, b):
    out = {}
    for k in a.keys():
        out[k] = {
            'sum': a[k]['sum'] + b[k]['sum'],
            'sumsq': a[k]['sumsq'] + b[k]['sumsq'],
            'min': min(a[k]['min'], b[k]['min']),
            'max': max(a[k]['max'], b[k]['max']),
            'count': a[k]['count'] + b[k]['count']
        }
    return out


def finalize_stats(agg):
    stats = {}
    for k, v in agg.items():
        if v['count'] == 0:
            continue
        mean = v['sum'] / v['count']
        var = max(v['sumsq'] / v['count'] - mean * mean, 0.0)
        std = np.sqrt(var)
        stats[k] = {
            'mean': float(mean),
            'std': float(std + 1e-8),
            'min': float(v['min']),
            'max': float(v['max']),
            'count': int(v['count'])
        }
    return stats


def process_file(path: str, keys: list, target_time_step: float, num_historical_steps: int):
    """
    子进程处理单个 .pkl 文件：
    - 加载文件
    - 逐场景预处理（重采样，局部化）
    - 累加局部特征的增量统计
    返回 (local_agg, local_count)
    """
    local_agg = init_aggregator(keys)
    local_count = 0
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return local_agg, local_count

    local_pre = MaritimeScenePreprocessor(target_time_step=target_time_step,
                                          num_historical_steps=num_historical_steps,
                                          verbose=False)
    num_scenes = len(data) if hasattr(data, '__len__') else 0
    for i in range(num_scenes):
        scene = data[i]
        processed = local_pre.preprocess_scene(scene)
        for ship in processed['ships']:
            feats = ship['features_local']
            values = {}
            for k in keys:
                if k in feats:
                    values[k] = np.asarray(feats[k]).reshape(-1)
            update_agg(local_agg, values)
        local_count += 1
    return local_agg, local_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='per_file 目录路径（包含多个 *_processed_batches.pkl）')
    parser.add_argument('--out', type=str, required=True,
                        help='输出的全局归一化统计 JSON 路径')
    parser.add_argument('--max_scenes', type=int, default=-1,
                        help='可选：限制遍历的场景数，-1 表示全部')
    parser.add_argument('--time_step', type=float, default=30.0,
                        help='时间步长（秒），默认30.0秒（匹配原始AIS数据）')
    parser.add_argument('--num_hist', type=int, default=5,
                        help='历史步数，默认5步')
    args = parser.parse_args()

    # 关闭详细打印，加快遍历
    pre = MaritimeScenePreprocessor(target_time_step=args.time_step, 
                                    num_historical_steps=args.num_hist,
                                    verbose=False)

    # 需要统计的特征键（局部坐标系）
    keys = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'theta', 'omega']
    # 并行友好的增量统计器
    agg_global = init_aggregator(keys)

    count = 0
    files = list_pkl_files(args.data_root)

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futures = []
        for fname in files:
            futures.append(ex.submit(process_file, os.path.join(args.data_root, fname), keys, args.time_step, args.num_hist))

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Files', unit='file'):
            local_agg, local_count = fut.result()
            agg_global = merge_aggs(agg_global, local_agg)
            count += local_count
            if args.max_scenes > 0 and count >= args.max_scenes:
                break

    # 计算全局 mean/std（由增量统计推得）
    stats = finalize_stats(agg_global)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"写入全局归一化统计: {args.out}")


if __name__ == '__main__':
    main()


