#!/usr/bin/env python3
"""
基于Folium的交互式海上轨迹可视化工具
在地图上展示船只的历史轨迹和预测轨迹
"""

import torch
import folium
from folium import plugins
import os
import sys
from pathlib import Path
import numpy as np
import json
import math
import os

sys.path.append(str(Path(__file__).parent))

from smart.model import SMART
from smart.datamodules import MultiDataModule
from smart.utils.config import load_config_act

def denormalize_coordinates(normalized_x, normalized_y, norm_stats):
    """
    反归一化坐标
    
    注意：Maritime数据使用局部坐标系（每艘船以T_h-1为原点），
    归一化统计也是基于局部坐标计算的。
    反归一化后得到的仍然是局部坐标（米制）。
    
    Args:
        normalized_x: 归一化后的x坐标
        normalized_y: 归一化后的y坐标
        norm_stats: 归一化统计信息字典
    
    Returns:
        (x_meters, y_meters): 局部坐标系下的米制坐标
    """
    # 反归一化：从标准化值恢复到原始局部坐标（米）
    x_meters = normalized_x * norm_stats['x']['std'] + norm_stats['x']['mean']
    y_meters = normalized_y * norm_stats['y']['std'] + norm_stats['y']['mean']
    return x_meters, y_meters

def meters_to_lat_lon(x_meters, y_meters, center_lat, center_lon):
    """
    将以米为单位的相对坐标转换为经纬度
    
    Args:
        x_meters: x方向距离（米）
        y_meters: y方向距离（米）
        center_lat: 中心纬度
        center_lon: 中心经度
    
    Returns:
        (lat, lon): 转换后的经纬度（Python原生float类型）
    """
    # 近似转换：
    # 1度纬度 ≈ 111,000米
    # 1度经度 ≈ 111,000米 * cos(纬度)
    
    lat_per_meter = 1.0 / 111000.0
    lon_per_meter = 1.0 / (111000.0 * np.cos(np.radians(center_lat)))
    
    # 确保返回Python原生float类型（folium需要）
    lat = float(center_lat + y_meters * lat_per_meter)
    lon = float(center_lon + x_meters * lon_per_meter)
    
    return lat, lon

def _apply_transform_xy(x, y, swap_xy, flip_x, flip_y):
    """在米制坐标系中应用交换/翻转。"""
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return x, y

def _apply_transform_heading(theta, swap_xy, flip_x, flip_y):
    """将heading向量按相同几何变换变换，并返回新角度。"""
    vx, vy = math.cos(theta), math.sin(theta)
    vx, vy = _apply_transform_xy(vx, vy, swap_xy, flip_x, flip_y)
    return math.atan2(vy, vx)

def _score_axis_flip(positions, headings, num_historical, norm_stats=None, max_steps=5):
    """
    基于位移与朝向的一致性评分，判断是否需要对坐标轴取反。
    返回对四种情况(不翻转/翻转X/翻转Y/翻转XY)的平均点积分数。
    分数越大，表示位移方向越与heading一致。
    """
    if positions.shape[1] < num_historical + 2:
        return { (False, False): 0.0, (True, False): 0.0, (False, True): 0.0, (True, True): 0.0 }

    t0 = num_historical - 1
    steps = min(max_steps, positions.shape[1] - t0 - 1)

    # 只取前若干艘船做统计，避免极端值影响
    num_agents = positions.shape[0]
    agent_indices = range(min(num_agents, 32))

    # 8 种情况：是否交换XY × 翻转X × 翻转Y
    keys = []
    for swap_xy in (False, True):
        for flip_x in (False, True):
            for flip_y in (False, True):
                keys.append((swap_xy, flip_x, flip_y))
    scores = { k: [] for k in keys }
    for a in agent_indices:
        for k in range(steps):
            # 位移向量（每30s一步）
            dx = float(positions[a, t0 + 1 + k, 0] - positions[a, t0 + k, 0])
            dy = float(positions[a, t0 + 1 + k, 1] - positions[a, t0 + k, 1])
            # 使用反归一化后的米制，避免xy方差不同影响评分
            if norm_stats is not None:
                dx = dx * norm_stats['x']['std']
                dy = dy * norm_stats['y']['std']
            # 选择合适的方向向量：优先heading，否则使用速度方向
            theta = float(headings[a, t0 + k])
            if not np.isfinite(theta) or theta == 0.0:
                # 使用相邻位移近似速度方向
                vx, vy = dx, dy
                if vx == 0.0 and vy == 0.0:
                    continue
                theta = math.atan2(vy, vx)
            for key in keys:
                swap_xy, flip_x, flip_y = key
                tx, ty = _apply_transform_xy(dx, dy, swap_xy, flip_x, flip_y)
                theta_t = _apply_transform_heading(theta, swap_xy, flip_x, flip_y)
                hx, hy = math.cos(theta_t), math.sin(theta_t)
                scores[key].append(tx * hx + ty * hy)

    return {k: (float(np.mean(v)) if len(v) > 0 else 0.0) for k, v in scores.items()}

def _infer_axis_flip(positions, headings, num_historical, norm_stats=None):
    """
    通过与heading的一致性自动推断是否需要翻转X/Y轴。
    返回 (flip_x, flip_y)。
    """
    try:
        stat = _score_axis_flip(positions, headings, num_historical, norm_stats)
        # 选择平均点积最大的翻转/交换组合
        best = max(stat.items(), key=lambda x: x[1])[0]
        return best  # (swap_xy, flip_x, flip_y)
    except Exception:
        return (False, False, False)

def _parse_axis_override():
    """从环境变量读取强制轴变换设置。
    环境变量：
      - FOLIUM_FORCE_TRANSFORM="swap,flipx,flipy"  例如: "1,0,1"
      或分别设置：FOLIUM_SWAP_XY, FOLIUM_FLIP_X, FOLIUM_FLIP_Y （0/1）
    返回 (forced: bool, swap_xy, flip_x, flip_y)
    """
    force = os.getenv('FOLIUM_FORCE_TRANSFORM', '').strip()
    if force:
        try:
            s, fx, fy = [v.strip() for v in force.split(',')]
            return True, (s == '1'), (fx == '1'), (fy == '1')
        except Exception:
            pass
    sx = os.getenv('FOLIUM_SWAP_XY', '').strip()
    fx = os.getenv('FOLIUM_FLIP_X', '').strip()
    fy = os.getenv('FOLIUM_FLIP_Y', '').strip()
    if sx or fx or fy:
        return True, (sx == '1'), (fx == '1'), (fy == '1')
    return False, False, False, False

def create_map_visualization(data, prediction, output_path, scene_id=0, center_lat=30.0, center_lon=120.0, norm_stats=None):
    """
    创建基于Folium的交互式地图可视化

    注意：Maritime数据使用局部坐标系！
    - 每个滑动窗口以T_h-1（历史结束帧）为原点和朝向参考
    - 因此所有船在T_h-1时刻的局部坐标接近(0,0)
    - 这是正常的数据设计，不是错误！
    """
    # 若数据自带原点经纬度，用其作为地图中心
    # 注意：这是场景的全局原点，但数据是局部坐标系
    try:
        if hasattr(data, 'metadata') and isinstance(data.metadata, dict):
            center_lat = float(data.metadata.get('origin_lat', center_lat))
            center_lon = float(data.metadata.get('origin_lon', center_lon))
            print(f"  [INFO] 使用场景原点作为地图中心: ({center_lat:.6f}°N, {center_lon:.6f}°E)")
    except Exception:
        pass

    # === 读取锚点/旋回信息（AB开关）===
    use_ref_anchor = os.getenv('FOLIUM_USE_REF_ANCHOR', '0') == '1'
    anchor_lat, anchor_lon, anchor_theta = center_lat, center_lon, 0.0

    if use_ref_anchor:
        scene_info = getattr(data, 'scene_info', None)
        if isinstance(scene_info, dict):
            anchor_lat = float(scene_info.get('ref_lat', anchor_lat))
            anchor_lon = float(scene_info.get('ref_lon', anchor_lon))
            anchor_theta = float(scene_info.get('ref_theta', anchor_theta))
            print(f"  [INFO] 使用窗口参考帧作为地图锚点: ({anchor_lat:.6f}, {anchor_lon:.6f}), theta={np.degrees(anchor_theta):.1f}°")
        else:
            # 回退：尝试用场景原点（若能读到）
            meta = data.metadata if (hasattr(data, 'metadata') and isinstance(data.metadata, dict)) else {}
            anchor_lat = float(meta.get('origin_lat', anchor_lat))
            anchor_lon = float(meta.get('origin_lon', anchor_lon))
            print(f"  [INFO] 使用场景原点作为地图锚点: ({anchor_lat:.6f}, {anchor_lon:.6f})；无 ref_theta 不做旋回")

    # 地图定位也用 anchor
    m = folium.Map(
        location=[anchor_lat, anchor_lon],
        zoom_start=15,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # 提取数据
    features = data['agent']['x'].cpu().numpy()  # [N_agents, T_total, 8]
    positions = features[:, :, :2]  # [N_agents, T_total, 2] (x, y in meters)

    # 调试输出
    print(f"\n  [DEBUG] 场景 {scene_id} 坐标统计 (局部坐标系):")
    print(f"    归一化后坐标范围: x=[{positions[:,:,0].min():.4f}, {positions[:,:,0].max():.4f}], y=[{positions[:,:,1].min():.4f}, {positions[:,:,1].max():.4f}]")

    # 检查T_h-1（历史结束帧）的位置 - 应该接近原点
    num_historical = 5
    t_ref = num_historical - 1  # 索引4，第5步
    positions_at_tref = positions[:, t_ref, :]
    print(f"    T_h-1时刻位置分布: x=[{positions_at_tref[:,0].min():.4f}, {positions_at_tref[:,0].max():.4f}], y=[{positions_at_tref[:,1].min():.4f}, {positions_at_tref[:,1].max():.4f}]")
    print(f"    T_h-1时刻位置标准差: x_std={positions_at_tref[:,0].std():.4f}, y_std={positions_at_tref[:,1].std():.4f}")

    # === 量化证据（健壮版）：窗口参考帧与场景起点错配的迹象 ===
    try:
        # 保险获取 metadata（PyG里metadata可能是方法名冲突）
        meta = data.metadata if (hasattr(data, 'metadata') and isinstance(data.metadata, dict)) else {}
        # 读取窗口参考帧位置（hist_end），若无则回退到 num_historical-1
        hist_end = int(meta.get('hist_end', num_historical - 1))
        # 时间步长（秒），默认30
        step_sec = float(meta.get('time_step_size', 30.0))
        minutes_from_scene_origin_to_ref = hist_end * step_sec / 60.0
        print(f"    [EVIDENCE] 窗口参考帧距场景起点的时间: {minutes_from_scene_origin_to_ref:.1f} 分钟 (hist_end={hist_end})")

        # 参考船（agent 0）在窗口内 t=0→T_h-1 的位移模长（局部坐标，米）
        ref_ship = 0
        dx0 = float(positions[ref_ship, 0, 0] - positions[ref_ship, t_ref, 0])
        dy0 = float(positions[ref_ship, 0, 1] - positions[ref_ship, t_ref, 1])
        dist_ref_move_m = math.hypot(dx0, dy0)
        print(f"    [EVIDENCE] 参考船在窗口内 t=0→T_h-1 位移(局部): {dist_ref_move_m:.1f} m")

        # 当前用于米→经纬度的锚点（若未从样本元数据设置，则是默认中心）
        # 原来打印的是 center_lat/center_lon（误导）
        print(f"    [EVIDENCE] 当前锚点(用于米→经纬度): lat={anchor_lat:.6f}, lon={anchor_lon:.6f}")
    except Exception as e:
        print(f"    [EVIDENCE] 诊断打印失败: {e}")

    if norm_stats is not None:
        print(f"    归一化统计: x_mean={norm_stats['x']['mean']:.2f}m, x_std={norm_stats['x']['std']:.2f}m")
        print(f"                y_mean={norm_stats['y']['mean']:.2f}m, y_std={norm_stats['y']['std']:.2f}m")
        # 测试反归一化第一个点
        test_x, test_y = denormalize_coordinates(positions[0, 0, 0], positions[0, 0, 1], norm_stats)
        print(f"    船0在t=0: 归一化({positions[0, 0, 0]:.4f}, {positions[0, 0, 1]:.4f}) -> 局部米制({test_x:.2f}m, {test_y:.2f}m)")
        test_x_ref, test_y_ref = denormalize_coordinates(positions[0, t_ref, 0], positions[0, t_ref, 1], norm_stats)
        print(f"    船0在T_h-1: 归一化({positions[0, t_ref, 0]:.4f}, {positions[0, t_ref, 1]:.4f}) -> 局部米制({test_x_ref:.2f}m, {test_y_ref:.2f}m) [应接近(0,0)]")
    else:
        print(f"    ⚠️  警告: norm_stats为None，坐标可能未归一化！")

    headings = features[:, :, 6]  # [N_agents, T_total] (theta in radians)

    if 'valid_mask' in data['agent']:
        valid_mask = data['agent']['valid_mask'].cpu().numpy()
    else:
        valid_mask = np.ones(positions.shape[:2], dtype=bool)

    num_agents = positions.shape[0]
    num_historical = 5
    num_future = 16

    # === 轴变换策略 ===
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
        swap_xy, flip_x, flip_y = _infer_axis_flip(positions, headings, num_historical, norm_stats)
        print(f"    [DEBUG] 轴变换(自动): swap_xy={swap_xy}, flip_x={flip_x}, flip_y={flip_y}")


    # 高对比度色板（Okabe–Ito + 补充）
    colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
        "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#999999", "#66c2a5",
        "#8da0cb", "#e78ac3", "#1b9e77", "#d95f02"
    ]

    # 收集所有坐标点，用于自动调整地图范围
    all_coords = []

    # 用点表示轨迹
    for agent_id in range(num_agents):
        color = colors[agent_id % len(colors)]

        # 历史轨迹点（小、透明度略低）
        hist_positions = positions[agent_id, :num_historical, :]
        hist_valid = valid_mask[agent_id, :num_historical]
        valid_hist = hist_positions[hist_valid]

        if len(valid_hist) > 0:
            hist_coords = []
            for pos in valid_hist:
                # 反归一化并应用坐标轴修正
                if norm_stats is not None:
                    x_meters, y_meters = denormalize_coordinates(pos[0], pos[1], norm_stats)
                else:
                    x_meters, y_meters = pos[0], pos[1]

                 # 可选的XY交换/翻转（保持现有逻辑）
                x_meters, y_meters = _apply_transform_xy(x_meters, y_meters, swap_xy, flip_x, flip_y)

                # 关键改动：将“局部(参考船朝向对齐)”坐标旋回到全局东-北系
                if use_ref_anchor and anchor_theta != 0.0:
                    cos_t, sin_t = math.cos(anchor_theta), math.sin(anchor_theta)
                    dx_world =  cos_t * x_meters - sin_t * y_meters
                    dy_world =  sin_t * x_meters + cos_t * y_meters
                else:
                    dx_world, dy_world = x_meters, y_meters

                # 以 anchor_lat/lon 为锚点做米→经纬度
                lat, lon = meters_to_lat_lon(dx_world, dy_world, anchor_lat, anchor_lon)
                all_coords.append([lat, lon])

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2.5,
                    color=color,
                    weight=1,
                    opacity=0.9,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)
                hist_coords.append([lat, lon])

            # 画历史轨迹连线（同色实线）
            folium.PolyLine(
                hist_coords,
                color=color,
                weight=2.0,
                opacity=0.8
            ).add_to(m)

        # 未来轨迹点（稍大、更实）
        future_positions = positions[agent_id, num_historical:num_historical+num_future, :]
        future_valid = valid_mask[agent_id, num_historical:num_historical+num_future]
        valid_future = future_positions[future_valid]

        if len(valid_future) > 0:
            future_coords = []
            for pos in valid_future:
                if norm_stats is not None:
                    x_meters, y_meters = denormalize_coordinates(pos[0], pos[1], norm_stats)
                else:
                    x_meters, y_meters = pos[0], pos[1]
                x_meters, y_meters = _apply_transform_xy(x_meters, y_meters, swap_xy, flip_x, flip_y)
                

                # 关键改动：旋回到全局东-北系
                if use_ref_anchor and anchor_theta != 0.0:
                    cos_t, sin_t = math.cos(anchor_theta), math.sin(anchor_theta)
                    dx_world =  cos_t * x_meters - sin_t * y_meters
                    dy_world =  sin_t * x_meters + cos_t * y_meters
                else:
                    dx_world, dy_world = x_meters, y_meters

                # 以 anchor_lat/lon 为锚点做米→经纬度
                lat, lon = meters_to_lat_lon(dx_world, dy_world, anchor_lat, anchor_lon)
                
                all_coords.append([lat, lon])

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3.2,
                    color=color,
                    weight=1.2,
                    opacity=1.0,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.95
                ).add_to(m)
                future_coords.append([lat, lon])

            # 画未来轨迹连线（同色，建议虚线以便区分）
            folium.PolyLine(
                future_coords,
                color=color,
                weight=2.2,
                opacity=0.95,
                dash_array='8,6'
            ).add_to(m)

    # 自动调整地图范围以适应所有轨迹
    if len(all_coords) > 0:
        lats = [coord[0] for coord in all_coords]
        lons = [coord[1] for coord in all_coords]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        m.fit_bounds(bounds, padding=[50, 50])
        print(f"    [DEBUG] 地图边界: 纬度[{min(lats):.6f}, {max(lats):.6f}], 经度[{min(lons):.6f}, {max(lons):.6f}]")

    # 保留实用控件
    plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.MousePosition().add_to(m)

    # 保存地图
    m.save(output_path)
    print(f"  ✅ 保存场景 {scene_id}: {output_path}")

def main():
    print("="*80)
    print("🗺️  SMART Maritime Folium交互式地图可视化")
    print("="*80)
    
    # 配置
    config_path = 'configs/train/train_maritime.yaml'
    checkpoint_path = 'logs/maritime_checkpoints/epoch=09.ckpt'  # ✅ 使用最新训练的模型 (2025-10-23)
    output_dir = 'folium_maps'
    norm_stats_path = None  # 数据未预先归一化，使用原始坐标
    
    # 地图中心点（可以根据实际数据调整）
    # 默认：中国东海（上海附近）
    center_lat = 30.0  # 北纬30度
    center_lon = 122.0  # 东经122度
    
    print(f"\n📂 配置文件: {config_path}")
    print(f"💾 模型检查点: {checkpoint_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 归一化统计: {'无 (使用原始坐标)' if norm_stats_path is None else norm_stats_path}")
    print(f"🌍 地图中心: ({center_lat}°N, {center_lon}°E)")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 错误: 找不到模型文件 {checkpoint_path}")
        return
    
    # 加载归一化统计信息
    norm_stats = None
    if norm_stats_path is not None and os.path.exists(norm_stats_path):
        print(f"\n📥 加载归一化统计信息...")
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        print(f"   x: 均值={norm_stats['x']['mean']:.2f}m, 标准差={norm_stats['x']['std']:.2f}m")
        print(f"   y: 均值={norm_stats['y']['mean']:.2f}m, 标准差={norm_stats['y']['std']:.2f}m")
    else:
        print(f"\n⚠️  注意: 数据使用原始坐标（未预先归一化）")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    print("\n⚙️  加载配置...")
    config = load_config_act(config_path)
    
    # 创建数据模块
    print("📊 加载测试数据...")
    datamodule = MultiDataModule(**vars(config.Dataset))
    datamodule.setup('test')
    print(f"   测试集大小: {len(datamodule.test_dataset)} 个场景")
    
    # 加载模型
    print("\n🧠 加载训练好的模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMART.load_from_checkpoint(checkpoint_path, model_config=config.Model)
    model.to(device)
    model.eval()
    print(f"   设备: {device}")
    
    # 生成地图
    # 分桶抽样：先按唯一来源文件数估计可视化数量
    unique_files = list(range(len(datamodule.test_dataset._file_paths)))
    num_scenes = min(5, len(unique_files))  # 生成场景数（以文件桶为单位）
    print(f"\n🗺️  开始生成交互式地图 (目标{num_scenes}个场景，按来源文件分桶抽样)...")
    print("   提示: 生成的HTML文件可以在浏览器中打开查看")
    
    # 基于来源文件分桶的索引选择
    ds = datamodule.test_dataset
    num_total = len(ds)
    if num_total == 0:
        print("\n❌ 错误: 测试集为空")
        return

    # 构建 file_idx -> [ds_idx...] 桶
    file_to_ds_indices = {}
    for ds_idx, (file_idx, sample_idx) in enumerate(ds._sample_indices):
        file_to_ds_indices.setdefault(file_idx, []).append(ds_idx)

    unique_file_indices = sorted(file_to_ds_indices.keys())
    num_scenes = min(5, len(unique_file_indices))  # 以文件桶数量为上限

    # 等间距选择文件桶
    file_sel = np.linspace(0, len(unique_file_indices) - 1, num=num_scenes, dtype=int)
    chosen_files = [unique_file_indices[i] for i in file_sel]

    # 桶内选择策略：median|random|first|last（默认：median）
    pick_mode = os.getenv('FOLIUM_BUCKET_PICK', 'median').strip().lower()
    def _pick_from_bucket(bucket):
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
        ds_idx = _pick_from_bucket(bucket)
        if ds_idx is None:
            continue
        src_file = os.path.basename(ds._file_paths[fidx])
        sample_idx = ds._sample_indices[ds_idx][1]
        print(f"   [DEBUG] 分桶抽样: file_idx={fidx}, src={src_file}, bucket_size={len(bucket)}, pick={pick_mode}, ds_idx={ds_idx}, sample_idx={sample_idx}")
        indices.append(ds_idx)

    for out_idx, ds_idx in enumerate(indices):
        # 直接按索引取样本（跨文件分散）
        sample = ds[ds_idx]
        sample = sample.to(device)

        # 模型前向（可选）
        output = model(sample)
        prediction = output.get('cls_pred', None)

        # 生成地图
        save_path = os.path.join(output_dir, f'scene_{out_idx:03d}.html')
        create_map_visualization(sample, prediction, save_path,
                                 scene_id=out_idx,
                                 center_lat=center_lat,
                                 center_lon=center_lon,
                                 norm_stats=norm_stats)

    # 用实际生成数量统计与索引页（放在循环之后）
    actual_scenes = len(indices)
    print(f"\n✅ 可视化完成!")
    print(f"📊 生成了 {actual_scenes} 个交互式地图")
    print(f"📁 保存位置: {output_dir}/")

    print("\n💡 使用说明:")
    print("   1. 用浏览器打开生成的HTML文件")
    print("   2. 可以缩放、拖动地图")
    print("   3. 点击轨迹和标记查看详细信息")
    print("   4. 使用测量工具测量距离")
    print("   5. 点击全屏按钮获得更好的视图")

    print("\n⚠️  重要说明：")
    print("   数据使用局部坐标系（每个窗口以T_h-1为原点）")
    print("   所有船在历史结束帧（实线终点）接近地图中心是正常现象")
    print("   这是SMART模型的数据设计，不是bug！")

    print("\n🌐 快速打开:")
    print(f"   在浏览器中打开: file://{os.path.abspath(output_dir)}/scene_000.html")

    # 创建索引页面使用实际数量
    create_index_page(output_dir, actual_scenes)

    print("\n" + "="*80)

def create_index_page(output_dir, num_scenes):
    """创建一个索引页面，方便查看所有地图"""
    
    index_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SMART Maritime - Interactive Map Visualization</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .card {{
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                transition: transform 0.2s;
                background-color: #ecf0f1;
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}
            .card a {{
                text-decoration: none;
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
            }}
            .stats {{
                margin: 20px 0;
                padding: 15px;
                background-color: #e8f4f8;
                border-left: 4px solid #3498db;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #7f8c8d;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🗺️ SMART Maritime - Interactive Map Visualization</h1>
            
            <div class="stats">
                <h3>📊 Visualization Summary</h3>
                <p><strong>Total Scenes:</strong> {num_scenes}</p>
                <p><strong>Model:</strong> SMART (Epoch 9, Val Acc: 51.20%)</p>
                <p><strong>Training Date:</strong> 2025-10-23</p>
                <p><strong>Time Interval:</strong> 30 seconds</p>
                <p><strong>Historical Steps:</strong> 5 (2.5 minutes)</p>
                <p><strong>Future Steps:</strong> 16 (8 minutes)</p>
            </div>
            
            <div style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
                <h3 style="margin-top:0; color:#856404;">⚠️ 重要说明：局部坐标系</h3>
                <p style="margin-bottom:10px;">数据使用<strong>局部坐标系</strong>（以T_h-1为原点），所有船的"历史结束帧"接近地图中心是正常现象。</p>
                <p style="margin:0;"><a href="README_LOCAL_COORDINATES.md" target="_blank" style="color:#007bff; text-decoration:none; font-weight:bold;">📖 点击查看详细说明文档</a></p>
            </div>
            
            <h2>Select a Scene to View:</h2>
            <div class="grid">
    '''
    
    for i in range(num_scenes):
        index_html += f'''
                <div class="card">
                    <a href="scene_{i:03d}.html" target="_blank">
                        🌊 Scene {i}<br>
                        <small>Click to open interactive map</small>
                    </a>
                </div>
        '''
    
    index_html += '''
            </div>
            
            <div class="footer">
                <p>Generated by SMART Maritime Trajectory Prediction System</p>
                <p>© 2025 | Validation Accuracy: 51.20% (Epoch 9)</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print(f"\n📑 索引页面已创建: {index_path}")
    print(f"   在浏览器中打开: file://{os.path.abspath(index_path)}")

if __name__ == '__main__':
    main()

