#!/usr/bin/env python3
"""
批量导出海上场景的滑动窗口样本为 .pt 分片（train/val/test）。

示例：
  python scripts/export_windows.py \
    --data_root /home/mahexing/ais_data_process/scene_generation/DI-MTP/data/per_file \
    --out_root  /home/mahexing/SMART-main/data/maritime_windows_v1 \
    --global_norm /home/mahexing/SMART-main/data/maritime_global_norm_v2.json \
    --num_hist 11 --num_fut 40 --stride 1 --shard_size 512 --seed 2025
"""

import os
import sys
import json
import pickle
import random
import argparse
from glob import glob
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 允许从仓库根目录导入预处理器
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maritime_scene_preprocessor import MaritimeScenePreprocessor


def process_scene_worker(args_tuple):
    """Top-level worker for multiprocessing (picklable).
    Args tuple: (f, si, split_out_dir, num_hist, num_fut, stride, target_time_step, global_norm_path, scene_shard_size)
    行为：在子进程内直接切分并写盘，返回计数信息，避免将大量样本通过IPC返回。
    """
    f, si, split_out_dir, num_hist, num_fut, stride, target_time_step, global_norm_path, scene_shard_size = args_tuple
    try:
        data = pickle.load(open(f, "rb"))
        scene = data[si]
    except Exception:
        return {"num_samples": 0, "num_files": 0}
    try:
        try:
            import torch as _torch
            _torch.set_num_threads(1)
        except Exception:
            pass
        local_pre = MaritimeScenePreprocessor(
            target_time_step=target_time_step,
            num_historical_steps=num_hist,
            apply_global_norm=False,  # 对齐Waymo：不归一化
            global_norm_stats_path=global_norm_path,
            verbose=False,
        )
        processed = local_pre.preprocess_scene(scene)
        samples = local_pre.create_hetero_data_windows(processed, num_hist, num_fut, stride)
        if not samples:
            return {"num_samples": 0, "num_files": 0}

        # 分片写盘（每个场景在子进程内写入多个小文件）
        os.makedirs(split_out_dir, exist_ok=True)
        total = 0
        num_files = 0
        chunk = []
        pid = os.getpid()
        base = os.path.splitext(os.path.basename(f))[0]
        part = 0
        for d in samples:
            chunk.append(d)
            total += 1
            if len(chunk) >= scene_shard_size:
                outp = os.path.join(split_out_dir, f"scene_{base}_idx{si}_pid{pid}_part{part:04d}.pt")
                torch.save(chunk, outp)
                chunk.clear()
                num_files += 1
                part += 1
        if chunk:
            outp = os.path.join(split_out_dir, f"scene_{base}_idx{si}_pid{pid}_part{part:04d}.pt")
            torch.save(chunk, outp)
            chunk.clear()
            num_files += 1
        return {"num_samples": total, "num_files": num_files}
    except Exception:
        return {"num_samples": 0, "num_files": 0}

def list_scenes(data_root: str) -> List[Tuple[str, int]]:
    files = sorted(glob(os.path.join(data_root, "*.pkl")))
    scenes: List[Tuple[str, int]] = []
    for f in files:
        try:
            n = len(pickle.load(open(f, "rb")))
            scenes += [(f, i) for i in range(n)]
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    return scenes


def split_indices(items: List[Tuple[str, int]], train_ratio: float, val_ratio: float, seed: int):
    items = items.copy()
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def export_split(split_name: str,
                 items: List[Tuple[str, int]],
                 out_dir: str,
                 pre: MaritimeScenePreprocessor,
                 num_hist: int, num_fut: int, stride: int,
                 shard_size: int,
                 workers: int = 1,
                 target_time_step: Optional[float] = None,
                 global_norm_path: Optional[str] = None,
                 scene_shard_size: int = 128) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    buf = []
    shard_idx = 0
    total = 0

    # 并行版：每个场景由子进程处理后返回样本列表，主进程负责分片写盘
    def _worker(args_tuple):
        f, si, num_hist, num_fut, stride, global_norm = args_tuple
        try:
            data = pickle.load(open(f, "rb"))
            scene = data[si]
        except Exception:
            return []
        try:
            local_pre = MaritimeScenePreprocessor(
                target_time_step=pre.target_time_step,
                num_historical_steps=num_hist,
                apply_global_norm=False,  # 对齐Waymo：不归一化,
                global_norm_stats_path=global_norm,
                verbose=False,
            )
            processed = local_pre.preprocess_scene(scene)
            samples = local_pre.create_hetero_data_windows(processed, num_hist, num_fut, stride)
            return samples
        except Exception:
            return []

    if workers <= 1:
        pbar = tqdm(items, desc=f"{split_name}", unit="scene")
        for f, si in pbar:
            # 串行模式：保持原聚合到大shard的逻辑
            try:
                data = pickle.load(open(f, "rb"))
                scene = data[si]
            except Exception as e:
                pbar.write(f"跳过场景 {f}[{si}]: {e}")
                continue
            try:
                processed = pre.preprocess_scene(scene)
                samples = pre.create_hetero_data_windows(processed, num_hist, num_fut, stride)
            except Exception as e:
                pbar.write(f"场景处理失败 {f}[{si}]: {e}")
                continue
            if not samples:
                continue
            sbar = tqdm(total=len(samples), desc=f"{split_name}-samples", unit="sample", leave=False)
            for d in samples:
                buf.append(d); total += 1; sbar.update(1)
                if len(buf) >= shard_size:
                    outp = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
                    torch.save(buf, outp); buf.clear(); shard_idx += 1
                    pbar.set_postfix(samples=total, shards=shard_idx)
            sbar.close()
    else:
        tt = target_time_step if target_time_step is not None else pre.target_time_step
        tasks = [(f, si, out_dir, num_hist, num_fut, stride, tt, global_norm_path, scene_shard_size)
                 for (f, si) in items]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_scene_worker, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}", unit="scene"):
                meta = fut.result() or {"num_samples": 0, "num_files": 0}
                total += int(meta.get("num_samples", 0))

    if workers <= 1 and buf:
        outp = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(buf, outp)
        shard_idx += 1

    # 在并行模式下，shard 由子进程写入；这里无法精确统计 shard 数，可统计目录中文件个数
    if workers > 1:
        shard_idx = len([n for n in os.listdir(out_dir) if n.endswith('.pt')])
    return {"num_scenes": len(items), "num_shards": shard_idx, "num_samples": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="per_file 目录（包含多个 *_processed_batches.pkl）")
    ap.add_argument("--out_root", type=str, required=True,
                    help="导出目录（将创建 train/val/test 子目录）")
    ap.add_argument("--global_norm", type=str, default=None,
                    help="全局归一化 JSON 文件路径（可选；已禁用归一化时可不传）")
    ap.add_argument("--num_hist", type=int, default=11)
    ap.add_argument("--num_fut", "--num_future", dest="num_fut", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=4, help="并行进程数，1 表示串行")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    test_ratio = max(0.0, 1.0 - (args.train_ratio + args.val_ratio))
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio 之和不能超过 1.0")

    # 列出全部场景并划分
    scenes = list_scenes(args.data_root)
    splits = split_indices(scenes, args.train_ratio, args.val_ratio, args.seed)

    # 预处理器（启用全局归一化）
    # 修改为30秒间隔以匹配原始AIS数据（避免不必要的插值）
    pre = MaritimeScenePreprocessor(target_time_step=30.0,
                                    num_historical_steps=args.num_hist,
                                    apply_global_norm=False,  # 对齐Waymo：不归一化,
                                    global_norm_stats_path=args.global_norm,
                                    verbose=False)

    # 导出
    os.makedirs(args.out_root, exist_ok=True)
    stats = {}
    for split in ["train", "val", "test"]:
        out_dir = os.path.join(args.out_root, split)
        s = export_split(split, splits[split], out_dir, pre,
                         num_hist=args.num_hist, num_fut=args.num_fut,
                         stride=args.stride, shard_size=args.shard_size,
                         workers=max(1, int(args.workers)),
                         target_time_step=pre.target_time_step,
                         global_norm_path=args.global_norm)
        stats[split] = s

    # 保存统计
    stats_path = os.path.join(args.out_root, "export_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("导出完成:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()


