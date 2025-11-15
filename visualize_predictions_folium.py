#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
import numpy as np
import torch
import folium
from folium.plugins import TimestampedGeoJson

from torch_geometric.loader import DataLoader
from smart.utils.config import load_config_act
from smart.model import SMART
from smart.datasets.scalable_dataset import MultiDataset
from smart.datasets.maritime_dataset import MaritimeDataset
from smart.transforms import WaymoTargetBuilder, MaritimeTargetBuilder

# 复用既有可视化脚本中的工具函数（坐标反归一化 / 米->经纬度 / 轴交换翻转 / 开关解析）
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
    # 默认启用参考锚点（可通过环境变量显式关闭：FOLIUM_USE_REF_ANCHOR=0）
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
        src_file = os.path.basename(ds._file_paths[fidx])
        sample_idx = ds._sample_indices[ds_idx][1]
        print(f"   [DEBUG] 分桶抽样: file_idx={fidx}, src={src_file}, bucket_size={len(bucket)}, pick={pick_mode}, ds_idx={ds_idx}, sample_idx={sample_idx}")
        indices.append(ds_idx)
    return indices

def _pick_indices_uniform(ds, num_scenes: int):
    """从整个数据集中按等间距选择样本索引。"""
    total = len(ds)
    if total <= 0:
        return []
    num = int(min(num_scenes, total))
    return np.linspace(0, total - 1, num=num, dtype=int).tolist()

def _pick_indices_random(ds, num_scenes: int, seed: int = 0):
    """从整个数据集中随机不放回抽样样本索引。"""
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

def _compute_ade_fde(gt_pos: np.ndarray, pred_pos: np.ndarray, valid_mask: np.ndarray = None):
    """
    计算 ADE/FDE。
    Args:
        gt_pos: [N, T, 2] GT未来坐标（局部米制）
        pred_pos: [N, T, 2] 预测坐标（局部米制）
        valid_mask: [N, T] bool，有效步掩码；None 则视为全 True
    Returns:
        scene_ade, scene_fde, ade_per_agent[N], fde_per_agent[N]
    """
    assert gt_pos.shape == pred_pos.shape, "gt 与 pred 形状不一致"
    N, T, _ = gt_pos.shape
    if valid_mask is None:
        valid_mask = np.ones((N, T), dtype=bool)

    # [N, T] 的逐步欧氏距离
    distances = np.linalg.norm(pred_pos - gt_pos, axis=-1)

    ade_per_agent = []
    fde_per_agent = []
    for i in range(N):
        m = valid_mask[i]
        if np.any(m):
            ade_per_agent.append(float(np.mean(distances[i][m])))
            # FDE：优先取最后一步；若最后一步无效，则回退到该agent最后一个有效步
            if m[-1]:
                fde_per_agent.append(float(distances[i, -1]))
            else:
                idxs = np.where(m)[0]
                fde_per_agent.append(float(distances[i, idxs[-1]]))
        else:
            ade_per_agent.append(np.nan)
            fde_per_agent.append(np.nan)

    ade_arr = np.asarray(ade_per_agent, dtype=float)
    fde_arr = np.asarray(fde_per_agent, dtype=float)
    scene_ade = float(np.nanmean(ade_arr))
    scene_fde = float(np.nanmean(fde_arr))
    return scene_ade, scene_fde, ade_arr, fde_arr

def _draw_scene_prediction_map(
    data, pred, output_path, scene_id, center_lat, center_lon, norm_stats, save_map: bool = True,
    animate: bool = False, step_seconds: int = 30, animate_speed: float = 1.0,
    animate_marker_radius: float = 3.0
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

    # 提取数据
    feat = data['agent']['x'].cpu().numpy()         # [N, T, 8]
    pos   = feat[:, :, :2]                          # [N, T, 2]
    heads = feat[:, :, 6]                           # [N, T]
    if 'valid_mask' in data['agent']:
        valid_mask = data['agent']['valid_mask'].cpu().numpy()
    else:
        valid_mask = np.ones(pos.shape[:2], dtype=bool)

    num_agents = pos.shape[0]
    num_his = 5
    num_fut = 16

    colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
        "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#8da0cb", "#e78ac3", "#1b9e77", "#d95f02"
    ]

    all_coords = []

    # 逐 agent 绘制：历史(实线小点) + 未来GT(绿色虚线) + 预测(红色粗虚线)
    pred_traj = pred['pred_traj'].detach().cpu().numpy()  # [N, num_fut, 2]
    pred_vmask = pred.get('valid_mask', None)
    if pred_vmask is not None:
        pred_vmask = pred_vmask.detach().cpu().numpy()     # [N, num_fut]
    else:
        pred_vmask = np.ones((num_agents, num_fut), dtype=bool)

    # === 量化误差：ADE / FDE ===
    gt_future = pos[:, num_his:num_his+num_fut, :]                       # [N, num_fut, 2]
    gt_vmask  = valid_mask[:, num_his:num_his+num_fut]                   # [N, num_fut]
    vmask     = (gt_vmask & pred_vmask) if pred_vmask is not None else gt_vmask
    scene_ade, scene_fde, _, _ = _compute_ade_fde(gt_future, pred_traj, vmask)
    print(f"  [METRIC] Scene {scene_id}: ADE={scene_ade:.2f} m, FDE={scene_fde:.2f} m")

    # === 追加诊断：累计路程/末端位移/方向一致性 ===
    # 累计路程（逐步位移和）
    gt_steps  = gt_future[:, 1:, :] - gt_future[:, :-1, :]
    pr_steps  = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
    gt_step_l = np.linalg.norm(gt_steps, axis=-1)                           # [N, T-1]
    pr_step_l = np.linalg.norm(pr_steps, axis=-1)                           # [N, T-1]
    # 只统计有效步（去掉无效位置差）
    vmask_step = vmask[:, 1:] & vmask[:, :-1]
    gt_path    = np.where(vmask_step, gt_step_l, 0.0).sum(axis=1)          # [N]
    pr_path    = np.where(vmask_step, pr_step_l, 0.0).sum(axis=1)          # [N]
    print(f"  [METRIC]   PathLen(mean): GT={float(np.nanmean(gt_path)):.1f} m, Pred={float(np.nanmean(pr_path)):.1f} m")

    # 末端位移（首末差）
    def _last_valid(vec, mask_row):
        idx = np.where(mask_row)[0]
        if len(idx) == 0:
            return None
        return vec[idx[-1]]
    gt_first = gt_future[:, 0, :]                                           # [N,2]
    def _fallback_last(vec, mask_row):
        lv = _last_valid(vec, mask_row)
        return lv if lv is not None else vec[-1]
    gt_last  = np.stack([_fallback_last(gt_future[i], vmask[i]) for i in range(gt_future.shape[0])])
    pr_last  = np.stack([_fallback_last(pred_traj[i], vmask[i]) for i in range(pred_traj.shape[0])])
    gt_disp  = np.linalg.norm(gt_last - gt_first, axis=-1)                  # [N]
    pr_disp  = np.linalg.norm(pr_last - gt_first, axis=-1)                  # [N]
    print(f"  [METRIC]   EndDisp(mean): GT={float(np.nanmean(gt_disp)):.1f} m, Pred={float(np.nanmean(pr_disp)):.1f} m")

    # 方向一致性（余弦相似度，逐步）
    eps = 1e-8
    gt_unit = gt_steps / (gt_step_l[..., None] + eps)
    pr_unit = pr_steps / (pr_step_l[..., None] + eps)
    cos_sim = (gt_unit * pr_unit).sum(axis=-1)                              # [N, T-1]
    cos_sim = np.where(vmask_step, cos_sim, np.nan)
    print(f"  [METRIC]   DirCos(mean over valid steps): {float(np.nanmean(cos_sim)):.3f}")

    # 若启用动画，准备 TimestampedGeoJson 的 Feature 列表
    features_time = []
    base_iso = "2020-01-01T00:00:00"
    def _time_of(t):
        secs = int(t * step_seconds)
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        return f"2020-01-01T{hh:02d}:{mm:02d}:{ss:02d}"

    for agent_id in range(num_agents):
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
        # 把历史最后点作为首点，连上 T_h-1 -> T_h
        if hist_last_coord is not None and len(fut_coords) > 0:
            fut_coords = [hist_last_coord] + fut_coords
        if not animate:
            if len(fut_coords) > 1:
                folium.PolyLine(
                    fut_coords,
                    color=color,
                    weight=2.8,
                    opacity=0.85,
                    dash_array="6,4"
                ).add_to(m)

        # 预测（红色）
        pred_positions = pred_traj[agent_id, :num_fut, :]
        pred_valid = pred_vmask[agent_id, :num_fut]
        pred_coords = []
        for ok, (x, y) in zip(pred_valid.tolist(), pred_positions):
            if not ok:
                continue
            lat, lon = _transform_point(x, y, use_ref_anchor, anchor_theta, swap_xy, flip_x, flip_y, anchor_lat, anchor_lon, norm_stats)
            all_coords.append([lat, lon])
            pred_coords.append([lat, lon])
        # 把历史最后点作为首点，连上 T_h-1 -> 预测第1步
        if hist_last_coord is not None and len(pred_coords) > 0:
            pred_coords = [hist_last_coord] + pred_coords
        if not animate:
            if len(pred_coords) > 1:
                folium.PolyLine(
                    pred_coords,
                    color=color,
                    weight=4.0,
                    opacity=1.0,
                    dash_array="8,5"
                ).add_to(m)

        # 动画帧：逐时间步绘制点（仅预测）
        if animate:
            # 从历史末点开始计时 t=0，然后未来步依次 t=1..num_fut
            # 历史最后一个有效点
            if hist_last_coord is not None:
                features_time.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [hist_last_coord[1], hist_last_coord[0]]},
                    "properties": {
                        "time": _time_of(0),
                        "style": {"color": color},
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.85,
                            "color": color,
                            "opacity": 1.0,
                            "weight": 1,
                            "radius": float(animate_marker_radius)
                        },
                        "icon": "circle",
                        "popup": f"Agent {agent_id} | t=H-1"
                    }
                })
            # 未来预测帧
            for t, ok in enumerate(pred_valid.tolist(), start=1):
                if not ok:
                    continue
                latlon = pred_coords[t] if hist_last_coord is not None else pred_coords[t-1]
                features_time.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [latlon[1], latlon[0]]},
                    "properties": {
                        "time": _time_of(t),
                        "style": {"color": color},
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.85,
                            "color": color,
                            "opacity": 1.0,
                            "weight": 1,
                            "radius": float(animate_marker_radius)
                        },
                        "icon": "circle",
                        "popup": f"Agent {agent_id} | t=+{t}"
                    }
                })
                # 连线：上一帧 -> 当前帧（随时间推进显示）
                if hist_last_coord is not None:
                    curr_idx = t
                    prev_idx = t - 1
                else:
                    curr_idx = t - 1
                    prev_idx = t - 2
                if prev_idx >= 0 and curr_idx < len(pred_coords):
                    prev_ll = pred_coords[prev_idx]
                    curr_ll = pred_coords[curr_idx]
                    features_time.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [prev_ll[1], prev_ll[0]],
                                [curr_ll[1], curr_ll[0]]
                            ]
                        },
                        "properties": {
                            "time": _time_of(t),
                            "style": {
                                "color": color,
                                "opacity": 0.9,
                                "weight": 2.5
                            },
                            "popup": f"Agent {agent_id} | seg t={t-1}->{t}"
                        }
                    })
               

    # 视野自适应
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # 如果启用动画，将时序要素加入地图
    if animate and features_time:
        # 将播放速度转化为更小的 period（步长：越小播放越快）
        eff_step_seconds = step_seconds
        try:
            if animate_speed and animate_speed > 0:
                eff = max(1, int(round(step_seconds / float(animate_speed))))
                eff_step_seconds = eff
        except Exception:
            eff_step_seconds = step_seconds
        tg = TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features_time
            },
            period=f"PT{int(eff_step_seconds)}S",
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options="YYYY-MM-DD HH:mm:ss",
            time_slider_drag_update=True
        )
        tg.add_to(m)

    if save_map:
        m.save(output_path)
        print(f"  ✅ 地图保存: {output_path}")

def _dump_token_selection(data, pred, output_json_path):
    try:
        nxt = pred.get('next_token_idx', None)
        prob = pred.get('pred_prob', None)
        nxt_gt = pred.get('next_token_idx_gt', None)
        eval_mask = pred.get('next_token_eval_mask', None)
        if nxt is None:
            print("  [WARN] 无 next_token_idx，跳过 token dump")
            return
        nxt = nxt.detach().cpu()
        num_agents, steps = nxt.shape[0], nxt.shape[1]
        obj = {
            'num_agents': int(num_agents),
            'num_steps': int(steps),
            'agents': []
        }
        # agent 类型（可选）
        agent_types = None
        try:
            if 'type' in data['agent']:
                agent_types = data['agent']['type'].detach().cpu().tolist()
        except Exception:
            agent_types = None
        for i in range(num_agents):
            rec = {
                'id': int(i),
                'picked_token_idx': [int(x) for x in nxt[i].tolist()]
            }
            if prob is not None:
                p = prob.detach().cpu()
                if p.ndim == 2 and p.shape[1] >= steps:
                    rec['picked_token_prob'] = [float(x) for x in p[i, :steps].tolist()]
                else:
                    rec['picked_token_prob'] = [float(x) for x in p[i].tolist()]
            if nxt_gt is not None:
                g = nxt_gt.detach().cpu()
                rec['gt_next_token_idx'] = [int(x) for x in g[i, :steps].tolist()]
            if eval_mask is not None:
                m = eval_mask.detach().cpu()
                rec['valid_mask'] = [bool(x) for x in m[i, :steps].tolist()]
            if agent_types is not None:
                rec['type'] = int(agent_types[i])
            obj['agents'].append(rec)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"  📝 token选择保存: {output_json_path}")
    except Exception as e:
        print(f"  [WARN] 保存 token 选择失败: {e}")

def _create_index_page(output_dir, num_scenes):
    html = ['''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMART Maritime - Prediction Visualization</title>
<style>
body{font-family:Arial,sans-serif;margin:20px;background:#f5f5f5;}
.container{max-width:1200px;margin:0 auto;background:#fff;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;margin-top:30px;}
.card{border:2px solid #e74c3c;border-radius:8px;padding:15px;text-align:center;transition:transform .2s;background:#fff8f8;}
.card:hover{transform:translateY(-4px);box-shadow:0 5px 15px rgba(0,0,0,0.2);}
.card a{text-decoration:none;color:#2c3e50;font-size:18px;font-weight:bold;}
.legend{margin:10px 0;color:#444}
</style></head><body><div class="container">
<h1>🗺️ SMART Maritime - Prediction Visualization</h1>
<div class="legend">图例：历史=原色实线；GT未来=绿色虚线；预测=红色虚线</div>
<div class="grid">''']
    for i in range(num_scenes):
        html.append(f'''
<div class="card"><a href="scene_{i:03d}.html" target="_blank">🌊 Scene {i}<br><small>Click to open</small></a></div>''')
    html.append('''</div></div></body></html>''')
    path = os.path.join(output_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(html))
    print(f"📑 索引已创建: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_maritime.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='folium_pred_maps')
    parser.add_argument('--bucket_pick', type=str, default=os.getenv('FOLIUM_BUCKET_PICK', 'median'),
                        choices=['median', 'random', 'first', 'last'])
    parser.add_argument('--norm_stats', type=str, default=os.getenv('FOLIUM_NORM_STATS', ''))
    parser.add_argument('--sample_mode', type=str, default=os.getenv('FOLIUM_SAMPLE_MODE', 'bucket'),
                        choices=['bucket', 'uniform', 'random'])
    parser.add_argument('--no_save_map', action='store_true')
    parser.add_argument('--dump_tokens', action='store_true', help='为每个场景导出 token 选择 JSON')
    parser.add_argument('--animate', action='store_true', help='启用时间动画（TimestampedGeoJson）')
    parser.add_argument('--step_seconds', type=int, default=30, help='每个预测步对应的秒数（默认30s）')
    parser.add_argument('--animate_speed', type=float, default=1.0, help='动画播放速度倍率（folium前端设置）')
    parser.add_argument('--animate_marker_radius', type=float, default=3.0, help='动画点半径（像素，默认3）')
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

    # 加载模型
    print("\n🧠 加载模型与权重 ...")
    model = SMART(config.Model)
    from smart.utils.log import Logging
    logger = Logging().log(level='INFO')
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

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

    print(f"\n🗺️  开始可视化预测(共 {len(indices)} 个场景) ...")
    for out_idx, ds_idx in enumerate(indices):
        sample = ds[ds_idx].to(device)
        with torch.no_grad():
            pred = model.inference(sample)

        save_path = os.path.join(args.output_dir, f'scene_{out_idx:03d}.html')
        _draw_scene_prediction_map(
            data=sample,
            pred=pred,
            output_path=save_path,
            scene_id=out_idx,
            center_lat=center_lat,
            center_lon=center_lon,
            norm_stats=norm_stats,
            save_map=not args.no_save_map,
            animate=args.animate,
            step_seconds=args.step_seconds,
            animate_speed=args.animate_speed,
            animate_marker_radius=args.animate_marker_radius
        )
        if args.dump_tokens:
            json_path = os.path.join(args.output_dir, f'scene_{out_idx:03d}_tokens.json')
            _dump_token_selection(sample, pred, json_path)

    if not args.no_save_map:
        _create_index_page(args.output_dir, len(indices))
    print("\n✅ 全部完成！")

if __name__ == '__main__':
    main()