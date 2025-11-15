#!/usr/bin/env python3
"""
海上场景预处理器
将原始AIS场景数据转换为SMART模型可用的格式
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import torch
from torch_geometric.data import HeteroData
from typing import Tuple, List, Dict, Any

class MaritimeScenePreprocessor:
    
    def __init__(self, target_time_step: float = 30.0, num_historical_steps: int = 5,
                 apply_global_norm: bool = False, global_norm_stats_path: str = None,
                 verbose: bool = True):
        # 注意：对齐Waymo - 不在预处理阶段归一化，保持米制单位
        """
        初始化预处理器
        
        Args:
            target_time_step: 目标时间步长(秒)，默认30.0秒（匹配原始AIS数据）
            num_historical_steps: 历史步数 T_h，用于定义局部坐标参考帧 t_ref = T_h-1
        """
        self.target_time_step = target_time_step
        self.num_historical_steps = num_historical_steps
        self.apply_global_norm = apply_global_norm
        self.global_norm_stats = None
        self.verbose = verbose
        if global_norm_stats_path:
            try:
                import json
                with open(global_norm_stats_path, 'r') as f:
                    self.global_norm_stats = json.load(f)
            except Exception:
                self.global_norm_stats = None
    
    def lonlat_to_meter(self, lon: np.ndarray, lat: np.ndarray, 
                       origin_lon: float, origin_lat: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        将经纬度坐标转换为以指定原点为基准的米制坐标
        
        Args:
            lon: 经度数组
            lat: 纬度数组  
            origin_lon: 原点经度
            origin_lat: 原点纬度
            
        Returns:
            (x, y): 米制坐标数组
        """
        # 地球半径 (米)
        R = 6371000.0
        
        # 转换为弧度
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        origin_lon_rad = np.radians(origin_lon)
        origin_lat_rad = np.radians(origin_lat)
        
        # 计算相对于原点的米制坐标
        # x = R * (lon - origin_lon) * cos(origin_lat)
        # y = R * (lat - origin_lat)
        x = R * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
        y = R * (lat_rad - origin_lat_rad)
        
        return x, y
    
    def resample_trajectory(self, traj_df: pd.DataFrame) -> pd.DataFrame:
        """
        将轨迹重采样到指定时间间隔
        
        Args:
            traj_df: 原始轨迹DataFrame
            
        Returns:
            重采样后的轨迹DataFrame
        """
        # 转换时间戳为秒数（相对于起始时间）
        start_time = traj_df['timestamp'].iloc[0]
        traj_df = traj_df.copy()
        traj_df['time_seconds'] = (traj_df['timestamp'] - start_time).dt.total_seconds()
        
        # 确定新的时间点
        max_time = traj_df['time_seconds'].max()
        # 确保至少包含原始数据的时间范围
        new_time_points = np.arange(0, max_time + 0.1, self.target_time_step)  # 添加小的缓冲
        
        # 对每个数值列进行插值
        interpolated_data = {'time_seconds': new_time_points}
        
        # 处理角度数据（航向角）需要特殊处理：对 cos/sin 分量插值，再用 atan2 合成
        angle_columns = []
        if 'cog' in traj_df.columns:
            angle_columns.append(('cog', 'cog_rad'))  # 单位：度（注意：将转换为数学角）
        if 'heading' in traj_df.columns:
            angle_columns.append(('heading', 'heading_rad'))  # 可选的额外角度列

        for deg_col, rad_out in angle_columns:
            angle_deg = traj_df[deg_col].values.astype(float)

            if deg_col == 'cog':
                # 航海COG: 北基(0°向北)、顺时针增
                # 数学角: 东基(0°向东)、逆时针增
                # 转换: theta_math = radians(90 - COG_deg)
                angle_math_rad = np.radians(90.0 - angle_deg)
                cos_val = np.cos(angle_math_rad)
                sin_val = np.sin(angle_math_rad)

                f_cos = interp1d(traj_df['time_seconds'].values, cos_val,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                f_sin = interp1d(traj_df['time_seconds'].values, sin_val,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')

                new_cos = f_cos(new_time_points)
                new_sin = f_sin(new_time_points)
                norm = np.hypot(new_cos, new_sin) + 1e-8
                new_cos /= norm
                new_sin /= norm

                new_theta_rad = np.arctan2(new_sin, new_cos)  # 数学角（东基、逆时针）
                new_theta_deg = (np.degrees(new_theta_rad) + 360) % 360
                # 反推回COG（若需要保持出参的一致性）：COG_deg = 90 - theta_deg
                new_cog_deg = (90.0 - new_theta_deg) % 360

                # 输出：提供 theta_rad（供下游使用），并保留 cog（度）
                interpolated_data['theta_rad'] = new_theta_rad
                interpolated_data['theta_deg'] = new_theta_deg
                interpolated_data['cog'] = new_cog_deg
                # 不再写入 cog_rad，避免误用
            else:
                # 其他角度（如heading），默认已为数学角（如不是请按需求转换）
                angle_rad = np.radians(angle_deg)
                cos_val = np.cos(angle_rad)
                sin_val = np.sin(angle_rad)

                f_cos = interp1d(traj_df['time_seconds'].values, cos_val,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                f_sin = interp1d(traj_df['time_seconds'].values, sin_val,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')

                new_cos = f_cos(new_time_points)
                new_sin = f_sin(new_time_points)
                norm = np.hypot(new_cos, new_sin) + 1e-8
                new_cos /= norm
                new_sin /= norm

                new_rad = np.arctan2(new_sin, new_cos)
                new_deg = (np.degrees(new_rad) + 360) % 360

                interpolated_data[deg_col] = new_deg
                interpolated_data[rad_out] = new_rad
        
        # 对其他数值列进行线性插值
        numeric_columns = ['lon', 'lat', 'sog']
        for col in numeric_columns:
            if col in traj_df.columns:
                f = interp1d(traj_df['time_seconds'].values, traj_df[col].values, 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated_data[col] = f(new_time_points)
        
        # 保留其他非数值列（取第一个值）
        for col in ['mmsi']:
            if col in traj_df.columns:
                interpolated_data[col] = [traj_df[col].iloc[0]] * len(new_time_points)
        
        # 重建时间戳
        interpolated_data['timestamp'] = [start_time + pd.Timedelta(seconds=t) for t in new_time_points]
        
        return pd.DataFrame(interpolated_data)
    
    def calculate_features(self, traj_df: pd.DataFrame, origin_lon: float, origin_lat: float) -> Dict[str, np.ndarray]:
        """
        计算轨迹的各种特征
        
        Args:
            traj_df: 重采样后的轨迹DataFrame
            origin_lon: 原点经度
            origin_lat: 原点纬度
            
        Returns:
            包含所有特征的字典
        """
        # 坐标转换
        x, y = self.lonlat_to_meter(traj_df['lon'].values, traj_df['lat'].values, 
                                   origin_lon, origin_lat)
        
        # 时间数组
        t = traj_df['time_seconds'].values
        dt = self.target_time_step
        
        # 速度计算 (m/s)
        # 首先从SOG转换 (1节 = 0.514444 m/s)
        sog_ms = traj_df['sog'].values * 0.514444
        
        # 🔧 修复：直接使用SOG和COG计算vx, vy（更准确）
        # COG/theta表示运动方向
        if 'theta_rad' in traj_df.columns:
            theta = traj_df['theta_rad'].values
        elif 'cog_rad' in traj_df.columns:
            theta = traj_df['cog_rad'].values
        else:
            theta = np.arctan2(np.gradient(y, dt), np.gradient(x, dt))
        
        # 从SOG和方向计算vx, vy
        vx = sog_ms * np.cos(theta)  # 使用SOG，而不是gradient
        vy = sog_ms * np.sin(theta)
        v_magnitude = sog_ms  # 直接使用SOG
        
        # 加速度计算 (m/s²)
        ax = np.gradient(vx, dt)  # x方向加速度
        ay = np.gradient(vy, dt)  # y方向加速度
        a_magnitude = np.sqrt(ax**2 + ay**2)  # 加速度大小
        
        # 航向角处理（优先使用数学角 theta_rad，其次退回 cog_rad，最后用速度方向）
        if 'theta_rad' in traj_df.columns:
            theta = traj_df['theta_rad'].values
        elif 'cog_rad' in traj_df.columns:
            theta = traj_df['cog_rad'].values
        else:
            # 从速度方向计算航向角
            theta = np.arctan2(vy, vx)
        
        # 航向变化率 (rad/s)
        omega = np.gradient(theta, dt)
        
        # 处理角度不连续性
        omega = np.where(np.abs(omega) > np.pi/dt, 
                        omega - np.sign(omega) * 2 * np.pi / dt, 
                        omega)
        
        features = {
            'time_seconds': t,
            'x': x,
            'y': y, 
            'theta': theta,           # 航向角 (rad)
            'theta_deg': np.degrees(theta),  # 航向角 (度)
            'vx': vx,                # x方向速度 (m/s)
            'vy': vy,                # y方向速度 (m/s) 
            'v_magnitude': v_magnitude,      # 速度大小 (m/s)
            'sog_ms': sog_ms,        # 原始SOG转换的速度 (m/s)
            'ax': ax,                # x方向加速度 (m/s²)
            'ay': ay,                # y方向加速度 (m/s²)
            'a_magnitude': a_magnitude,      # 加速度大小 (m/s²)
            'omega': omega,          # 航向变化率 (rad/s)
            'mmsi': traj_df['mmsi'].values[0]  # 船只标识
        }
        
        return features

    def wrap_angle_rad(self, angle: np.ndarray) -> np.ndarray:
        """将角度归一化到(-pi, pi]区间"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def to_local_at_hist_end(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        将全局特征转换到以历史结束帧 T_h-1 为原点与朝向的局部坐标系。
        使得在 t_ref (T_h-1) 时刻满足: x'=0, y'=0, theta'=0。
        """
        # 参考帧：历史结束帧
        t_ref = int(min(self.num_historical_steps - 1, len(features['time_seconds']) - 1))
        theta_ref = float(features['theta'][t_ref])
        cos_t = np.cos(theta_ref)
        sin_t = np.sin(theta_ref)

        # 平移至参考点
        dx = features['x'] - float(features['x'][t_ref])
        dy = features['y'] - float(features['y'][t_ref])

        # 旋转到参考朝向（使参考帧朝向为0）
        x_local =  cos_t * dx + sin_t * dy
        y_local = -sin_t * dx + cos_t * dy

        vx_local =  cos_t * features['vx'] + sin_t * features['vy']
        vy_local = -sin_t * features['vx'] + cos_t * features['vy']

        ax_local =  cos_t * features['ax'] + sin_t * features['ay']
        ay_local = -sin_t * features['ax'] + cos_t * features['ay']

        theta_local = self.wrap_angle_rad(features['theta'] - theta_ref)
        omega = features['omega']  # 常量偏移不改变角速度

        return {
            'time_seconds': features['time_seconds'],
            'x': x_local,
            'y': y_local,
            'vx': vx_local,
            'vy': vy_local,
            'ax': ax_local,
            'ay': ay_local,
            'v_magnitude': features.get('v_magnitude', np.sqrt(vx_local ** 2 + vy_local ** 2)),
            'a_magnitude': features.get('a_magnitude', np.sqrt(ax_local ** 2 + ay_local ** 2)),
            'theta': theta_local,
            'theta_deg': np.degrees(theta_local),
            'omega': omega,
            'mmsi': features['mmsi']
        }

    def to_local_at_t_ref(self, features: Dict[str, np.ndarray], t_ref: int) -> Dict[str, np.ndarray]:
        """
        基于任意参考帧 t_ref 将全局特征转换至局部坐标（x'=0,y'=0,theta'=0 at t_ref）。
        """
        t_ref = int(max(0, min(t_ref, len(features['time_seconds']) - 1)))
        theta_ref = float(features['theta'][t_ref])
        cos_t = np.cos(theta_ref)
        sin_t = np.sin(theta_ref)

        dx = features['x'] - float(features['x'][t_ref])
        dy = features['y'] - float(features['y'][t_ref])

        x_local =  cos_t * dx + sin_t * dy
        y_local = -sin_t * dx + cos_t * dy

        vx_local =  cos_t * features['vx'] + sin_t * features['vy']
        vy_local = -sin_t * features['vx'] + cos_t * features['vy']

        ax_local =  cos_t * features['ax'] + sin_t * features['ay']
        ay_local = -sin_t * features['ax'] + cos_t * features['ay']

        theta_local = self.wrap_angle_rad(features['theta'] - theta_ref)
        omega = features['omega']

        return {
            'time_seconds': features['time_seconds'],
            'x': x_local,
            'y': y_local,
            'vx': vx_local,
            'vy': vy_local,
            'ax': ax_local,
            'ay': ay_local,
            'v_magnitude': features.get('v_magnitude', np.sqrt(vx_local ** 2 + vy_local ** 2)),
            'a_magnitude': features.get('a_magnitude', np.sqrt(ax_local ** 2 + ay_local ** 2)),
            'theta': theta_local,
            'theta_deg': np.degrees(theta_local),
            'omega': omega,
            'mmsi': features['mmsi']
        }

    def generate_window_indices(self, total_steps: int, num_historical_steps: int, num_future_steps: int,
                                stride: int = 1) -> List[Dict[str, int]]:
        """
        生成滑动窗口的索引列表。
        返回的每个元素包含：hist_start, hist_end(=t_ref), fut_end（不含）
        """
        windows = []
        if total_steps < (num_historical_steps + num_future_steps):
            return windows
        # t_ref 是历史结束帧索引
        t_ref_start = num_historical_steps - 1
        t_ref_end = total_steps - num_future_steps - 1
        for t_ref in range(t_ref_start, t_ref_end + 1, stride):
            hist_start = t_ref - (num_historical_steps - 1)
            hist_end = t_ref
            fut_end = t_ref + num_future_steps + 1  # 右开区间
            windows.append({'hist_start': hist_start, 'hist_end': hist_end, 'fut_end': fut_end})
        return windows

    def create_hetero_data_for_window(self, processed_scene: Dict[str, Any],
                                      hist_start: int, hist_end: int, fut_end: int) -> HeteroData:
        """
        基于单个滑动窗口构建 HeteroData：
        - 使用局部坐标（参考帧 = hist_end，所有船相对于第一艘船）
        - 构造输入序列 = 历史 + 未来（总帧数 = fut_end - hist_start）
        - 额外提供 mask 与未来目标坐标，便于训练
        """
        ships = processed_scene['ships']
        impacts = processed_scene['impacts']
        ship_features_global = [ship['features_global'] for ship in ships]
        num_ships = len(ships)
        num_hist = hist_end - hist_start + 1
        num_future = fut_end - hist_end - 1
        total_steps = fut_end - hist_start

        # 🔧 修复：使用统一的参考点（第一艘船在hist_end的位置和朝向）
        # 而不是让每艘船以自己为原点！
        reference_ship = ship_features_global[0]  # av_index = 0
        t_ref = hist_end
        theta_ref = float(reference_ship['theta'][t_ref])
        x_ref = float(reference_ship['x'][t_ref])
        y_ref = float(reference_ship['y'][t_ref])
        cos_t = np.cos(theta_ref)
        sin_t = np.sin(theta_ref)

        # 计算窗口参考帧的真实经纬度（基于场景原点反推）
        R = 6371000.0
        origin_lat = float(processed_scene['metadata']['origin_lat'])
        origin_lon = float(processed_scene['metadata']['origin_lon'])
        origin_lat_rad = np.radians(origin_lat)

        ref_lat = origin_lat + np.degrees(y_ref / R)
        ref_lon = origin_lon + np.degrees(x_ref / (R * np.cos(origin_lat_rad)))

        # 计算局部特征（所有船相对于统一参考点）并裁剪到窗口区间
        agent_features = []
        target_xy = []
        is_history_mask = np.zeros((num_ships, total_steps), dtype=bool)
        for s in range(num_ships):
            feats_g = ship_features_global[s]
            
            # 转换到统一的局部坐标系（相对于参考船）
            dx = feats_g['x'] - x_ref
            dy = feats_g['y'] - y_ref
            x_local =  cos_t * dx + sin_t * dy
            y_local = -sin_t * dx + cos_t * dy
            
            vx_local =  cos_t * feats_g['vx'] + sin_t * feats_g['vy']
            vy_local = -sin_t * feats_g['vx'] + cos_t * feats_g['vy']
            
            ax_local =  cos_t * feats_g['ax'] + sin_t * feats_g['ay']
            ay_local = -sin_t * feats_g['ax'] + cos_t * feats_g['ay']
            
            theta_local = self.wrap_angle_rad(feats_g['theta'] - theta_ref)
            omega_local = feats_g['omega']  # 角速度不受坐标系平移和旋转影响
            
            # 切片窗口（使用新计算的局部坐标）
            x_arr = x_local[hist_start:fut_end]
            y_arr = y_local[hist_start:fut_end]
            vx_arr = vx_local[hist_start:fut_end]
            vy_arr = vy_local[hist_start:fut_end]
            ax_arr = ax_local[hist_start:fut_end]
            ay_arr = ay_local[hist_start:fut_end]
            theta_arr = theta_local[hist_start:fut_end]
            omega_arr = omega_local[hist_start:fut_end]

            # ========== 归一化已禁用（对齐Waymo）==========
            # Waymo不在预处理时归一化，保持米制单位(m, m/s)
            if False:  # self.apply_global_norm and self.global_norm_stats is not None:
                def norm_arr(key, arr):
                    stats = self.global_norm_stats.get(key, None)
                    if stats is None:
                        return arr
                    mean, std = float(stats['mean']), float(stats['std']) + 1e-8
                    return (arr - mean) / std
                x_arr  = norm_arr('x',  x_arr)
                y_arr  = norm_arr('y',  y_arr)
                vx_arr = norm_arr('vx', vx_arr)
                vy_arr = norm_arr('vy', vy_arr)
                ax_arr = norm_arr('ax', ax_arr)
                ay_arr = norm_arr('ay', ay_arr)
                theta_arr = norm_arr('theta', theta_arr)
                omega_arr = norm_arr('omega', omega_arr)

            ship_matrix = np.stack([x_arr, y_arr, vx_arr, vy_arr, ax_arr, ay_arr, theta_arr, omega_arr], axis=1)
            agent_features.append(ship_matrix)

            # 未来目标（未归一化，使用局部坐标原值，相对于统一参考点）
            target_xy.append(
                np.stack([
                    x_local[hist_end+1:fut_end],
                    y_local[hist_end+1:fut_end]
                ], axis=1)
            )
            is_history_mask[s, :num_hist] = True

        agent_features = np.stack(agent_features, axis=0)  # [N, total_steps, F]
        target_xy = np.stack(target_xy, axis=0)            # [N, num_future, 2]

        # 边关系（在全局坐标系的 hist_end 帧评估）
        edge_indices = self.build_graph_edges(ship_features_global, impacts, t_ref=hist_end)

        data = HeteroData()
        data['agent'].x = torch.tensor(agent_features, dtype=torch.float32)
        data['agent'].num_nodes = num_ships
        data['agent'].valid_mask = torch.ones(num_ships, total_steps, dtype=torch.bool)
        data['agent'].is_history_mask = torch.tensor(is_history_mask, dtype=torch.bool)
        data['agent'].target_xy = torch.tensor(target_xy, dtype=torch.float32)
        data['agent'].type = torch.zeros(num_ships, dtype=torch.long)
        data['agent'].av_index = torch.tensor(0, dtype=torch.long)
        data['agent'].mmsi = torch.tensor([ship['mmsi'] for ship in ships], dtype=torch.long)

        if edge_indices['interaction_edges'].numel() > 0:
            data['agent', 'interacts_with', 'agent'].edge_index = edge_indices['interaction_edges']
        if edge_indices['proximity_edges'].numel() > 0:
            data['agent', 'near_to', 'agent'].edge_index = edge_indices['proximity_edges']

        data.metadata = {
            **processed_scene['metadata'],
            'hist_start': hist_start,
            'hist_end': hist_end,
            'fut_end': fut_end,
            'num_hist': num_hist,
            'num_future': num_future
        }
        data.original_impacts = torch.tensor(impacts, dtype=torch.float32)

        data.scene_info = {
            'ref_lat': float(ref_lat),
            'ref_lon': float(ref_lon),
            'ref_theta': float(theta_ref)
        }

        return data

    def create_hetero_data_windows(self, processed_scene: Dict[str, Any],
                                   num_historical_steps: int, num_future_steps: int,
                                   stride: int = 1) -> List[HeteroData]:
        """
        为一个场景生成多个滑动窗口的 HeteroData 样本。
        """
        total_steps = processed_scene['metadata']['time_steps']
        windows = self.generate_window_indices(total_steps, num_historical_steps, num_future_steps, stride)
        samples = []
        for w in windows:
            data = self.create_hetero_data_for_window(processed_scene, w['hist_start'], w['hist_end'], w['fut_end'])
            samples.append(data)
        return samples
    
    def normalize_features(self, all_ship_features: List[Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """
        对所有船只的特征进行归一化
        
        Args:
            all_ship_features: 所有船只的特征列表
            
        Returns:
            (normalized_features, normalization_stats): 归一化后的特征和统计信息
        """
        # 说明：SMART范式要求在局部坐标系下工作，一般不对几何特征做全局Z-score。
        # 出于调试可视化之用，此函数保留，但不在HeteroData中应用其结果。
        # 收集所有船只的特征用于计算全局统计
        all_features = {}
        feature_keys = ['x', 'y', 'vx', 'vy', 'v_magnitude', 'ax', 'ay', 'a_magnitude', 'theta', 'omega']
        
        for key in feature_keys:
            all_values = []
            for ship_features in all_ship_features:
                if key in ship_features:
                    all_values.extend(ship_features[key].flatten())
            all_features[key] = np.array(all_values)
        
        # 计算归一化统计信息
        normalization_stats = {}
        for key, values in all_features.items():
            normalization_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) + 1e-8,  # 避免除零
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # 归一化每艘船的特征
        normalized_ships = []
        for ship_features in all_ship_features:
            normalized_ship = {}
            for key, values in ship_features.items():
                if key in normalization_stats:
                    # Z-score归一化
                    normalized_ship[key] = (values - normalization_stats[key]['mean']) / normalization_stats[key]['std']
                else:
                    # 非数值特征保持不变
                    normalized_ship[key] = values
            normalized_ships.append(normalized_ship)
        
        return normalized_ships, normalization_stats
    
    def build_graph_edges(self, ship_features_global: List[Dict[str, np.ndarray]], 
                         impacts: np.ndarray,
                         t_ref: int,
                         distance_threshold: float = 1000.0) -> Dict[str, torch.Tensor]:
        """
        构建图的边关系
        
        Args:
            ship_features_global: 船只全局特征列表(用于距离计算)
            impacts: 影响矩阵
            t_ref: 邻近评估参考帧（历史结束帧）
            distance_threshold: 距离阈值(米)
            
        Returns:
            边索引字典
        """
        num_ships = len(ship_features_global)
        
        # 基于impacts矩阵的交互边
        interaction_edges = []
        for i in range(num_ships):
            for j in range(num_ships):
                if i != j and impacts[i, j] > 0:
                    interaction_edges.append([i, j])
        
        # 基于距离的邻近边（仅在参考帧 t_ref 进行评估）
        proximity_edges = []
        for i in range(num_ships):
            for j in range(i + 1, num_ships):
                dx = ship_features_global[i]['x'][t_ref] - ship_features_global[j]['x'][t_ref]
                dy = ship_features_global[i]['y'][t_ref] - ship_features_global[j]['y'][t_ref]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < distance_threshold:
                    proximity_edges.append([i, j])
                    proximity_edges.append([j, i])  # 无向边
        
        # 去重邻近边
        proximity_edges = list(set([tuple(edge) for edge in proximity_edges]))
        proximity_edges = [list(edge) for edge in proximity_edges]
        
        edge_indices = {
            'interaction_edges': torch.tensor(interaction_edges).t().contiguous() if interaction_edges else torch.empty((2, 0), dtype=torch.long),
            'proximity_edges': torch.tensor(proximity_edges).t().contiguous() if proximity_edges else torch.empty((2, 0), dtype=torch.long)
        }
        
        return edge_indices
    
    def create_hetero_data(self, processed_scene: Dict[str, Any]) -> HeteroData:
        """
        创建PyTorch Geometric HeteroData结构
        
        Args:
            processed_scene: 预处理后的场景数据
            
        Returns:
            HeteroData对象
        """
        ships = processed_scene['ships']
        impacts = processed_scene['impacts']
        metadata = processed_scene['metadata']

        # 提取局部与全局特征
        ship_features_local = [ship['features_local'] for ship in ships]
        ship_features_global = [ship['features_global'] for ship in ships]

        # 构建节点特征张量（使用局部坐标系特征），并在需要时应用全局归一化
        num_ships = len(ships)
        num_timesteps = len(ship_features_local[0]['time_seconds'])

        agent_features = []
        for ship_data in ship_features_local:
            # ========== 归一化已禁用（对齐Waymo）==========
            # Waymo不在预处理时归一化，保持米制单位(m, m/s)
            if False:  # self.apply_global_norm and self.global_norm_stats is not None:
                # 使用固定的全局均值/标准差
                def norm_arr(key, arr):
                    stats = self.global_norm_stats.get(key, None)
                    if stats is None:
                        return arr
                    mean, std = float(stats['mean']), float(stats['std']) + 1e-8
                    return (arr - mean) / std
                x_arr  = norm_arr('x',  ship_data['x'])
                y_arr  = norm_arr('y',  ship_data['y'])
                vx_arr = norm_arr('vx', ship_data['vx'])
                vy_arr = norm_arr('vy', ship_data['vy'])
                ax_arr = norm_arr('ax', ship_data['ax'])
                ay_arr = norm_arr('ay', ship_data['ay'])
                theta_arr = norm_arr('theta', ship_data['theta'])
                omega_arr = norm_arr('omega', ship_data['omega'])
            else:
                x_arr, y_arr = ship_data['x'], ship_data['y']
                vx_arr, vy_arr = ship_data['vx'], ship_data['vy']
                ax_arr, ay_arr = ship_data['ax'], ship_data['ay']
                theta_arr, omega_arr = ship_data['theta'], ship_data['omega']
            ship_feature_matrix = np.stack([
                x_arr,
                y_arr, 
                vx_arr,
                vy_arr,
                ax_arr,
                ay_arr,
                theta_arr,
                omega_arr
            ], axis=1)
            agent_features.append(ship_feature_matrix)

        agent_features = np.stack(agent_features, axis=0)

        # 构建边关系：在全局坐标系的历史结束帧评估邻近关系
        t_ref = int(min(self.num_historical_steps - 1, num_timesteps - 1))
        edge_indices = self.build_graph_edges(ship_features_global, impacts, t_ref)
        
        # 创建HeteroData
        data = HeteroData()
        
        # Agent节点数据
        data['agent'].x = torch.tensor(agent_features, dtype=torch.float32)
        data['agent'].num_nodes = num_ships
        
        # Agent类型信息 (海上场景中所有都是船只，类型设为0)
        data['agent'].type = torch.zeros(num_ships, dtype=torch.long)
        
        # 有效性掩码 (简化：所有时间步都有效)
        data['agent'].valid_mask = torch.ones(num_ships, num_timesteps, dtype=torch.bool)
        
        # AV索引 (选择第一艘船作为关注对象)
        data['agent'].av_index = torch.tensor(0, dtype=torch.long)
        
        # 船只ID
        data['agent'].mmsi = torch.tensor([ship['mmsi'] for ship in ships], dtype=torch.long)
        
        # 边关系
        if edge_indices['interaction_edges'].numel() > 0:
            data['agent', 'interacts_with', 'agent'].edge_index = edge_indices['interaction_edges']
        
        if edge_indices['proximity_edges'].numel() > 0:
            data['agent', 'near_to', 'agent'].edge_index = edge_indices['proximity_edges']
        
        # 添加元数据
        data.metadata = metadata
        data.original_impacts = torch.tensor(impacts, dtype=torch.float32)
        
        return data
    
    def validate_hetero_data(self, data: HeteroData) -> Dict[str, Any]:
        """
        验证HeteroData结构
        
        Args:
            data: HeteroData对象
            
        Returns:
            验证结果字典
        """
        validation_results = {
            'structure_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # 验证节点数据
            if 'agent' in data:
                agent_data = data['agent']
                num_ships, num_timesteps, num_features = agent_data.x.shape
                
                validation_results['statistics'].update({
                    'num_ships': num_ships,
                    'num_timesteps': num_timesteps,
                    'num_features': num_features,
                    'agent_features_shape': list(agent_data.x.shape),
                    'agent_features_dtype': str(agent_data.x.dtype)
                })
                
                # 检查特征是否包含NaN或Inf
                if torch.isnan(agent_data.x).any():
                    validation_results['errors'].append("Agent features contain NaN values")
                    validation_results['structure_valid'] = False
                
                if torch.isinf(agent_data.x).any():
                    validation_results['errors'].append("Agent features contain Inf values")
                    validation_results['structure_valid'] = False
                
                # 检查valid_mask
                if hasattr(agent_data, 'valid_mask'):
                    if agent_data.valid_mask.shape != (num_ships, num_timesteps):
                        validation_results['errors'].append(f"Valid mask shape mismatch: {agent_data.valid_mask.shape} vs expected ({num_ships}, {num_timesteps})")
                        validation_results['structure_valid'] = False
                
            # 验证边数据
            edge_types = []
            for edge_type in data.edge_types:
                edge_data = data[edge_type]
                edge_index = edge_data.edge_index
                num_edges = edge_index.shape[1]
                edge_types.append({
                    'type': str(edge_type),
                    'num_edges': num_edges,
                    'shape': list(edge_index.shape)
                })
            
            validation_results['statistics']['edge_types'] = edge_types
            
            # 检查特征统计
            if 'agent' in data:
                features = data['agent'].x
                validation_results['statistics']['feature_stats'] = {
                    'mean': features.mean(dim=(0,1)).tolist(),
                    'std': features.std(dim=(0,1)).tolist(),
                    'min': features.min().item(),
                    'max': features.max().item()
                }
            
        except Exception as e:
            validation_results['structure_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def print_hetero_data_info(self, data: HeteroData):
        """打印HeteroData的详细结构信息"""
        print(f"\n🔍 HeteroData详细结构信息:")
        
        # 节点类型
        print(f"   节点类型: {list(data.node_types)}")
        
        # 边类型  
        if len(data.edge_types) > 0:
            print(f"   边类型: {list(data.edge_types)}")
        else:
            print(f"   边类型: 无")
        
        # Agent节点详情
        if 'agent' in data:
            agent_data = data['agent']
            print(f"\n   Agent节点详情:")
            for key, value in agent_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"     {key}: {value} ({type(value).__name__})")
        
        # 边详情
        for edge_type in data.edge_types:
            edge_data = data[edge_type]
            print(f"\n   边 {edge_type} 详情:")
            for key, value in edge_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"     {key}: {value} ({type(value).__name__})")
        
        # 附加属性
        extra_attrs = []
        for key in data.keys():
            if key not in data.node_types and key not in [str(et) for et in data.edge_types]:
                extra_attrs.append(key)
        
        if extra_attrs:
            print(f"\n   额外属性: {extra_attrs}")
    
    def preprocess_scene(self, scene_data: Tuple[List[pd.DataFrame], np.ndarray]) -> Dict[str, Any]:
        """
        预处理单个场景数据
        
        Args:
            scene_data: (trajectories, impacts) 元组
            
        Returns:
            预处理后的场景数据字典
        """
        trajectories, impacts = scene_data
        
        if self.verbose:
            print(f"开始预处理场景：{len(trajectories)}艘船只")
        
        # 确定原点：第一艘船的第一个时间点
        origin_traj = trajectories[0]
        origin_lon = origin_traj['lon'].iloc[0]
        origin_lat = origin_traj['lat'].iloc[0]
        origin_time = origin_traj['timestamp'].iloc[0]
        
        if self.verbose:
            print(f"原点设置: ({origin_lon:.6f}, {origin_lat:.6f}) at {origin_time}")
        
        # 处理每艘船的轨迹
        processed_ships = []
        all_features_global = []
        
        for i, traj in enumerate(trajectories):
            if self.verbose:
                print(f"处理船只 {i+1}/{len(trajectories)}: MMSI={traj['mmsi'].iloc[0]}")
            
            # 时间重采样
            resampled_traj = self.resample_trajectory(traj)
            
            # 特征计算（全局）
            features_global = self.calculate_features(resampled_traj, origin_lon, origin_lat)
            all_features_global.append(features_global)
            
            processed_ships.append({
                'ship_id': i,
                'mmsi': features_global['mmsi'],
                'features_global': features_global,
                'original_length': len(traj),
                'resampled_length': len(resampled_traj)
            })
        
        # 🔧 修复：使用统一的参考点（第一艘船）计算局部坐标
        # 而不是让每艘船以自己为原点
        if len(all_features_global) > 0:
            reference_ship = all_features_global[0]
            t_ref = min(self.num_historical_steps - 1, len(reference_ship['time_seconds']) - 1)
            theta_ref = float(reference_ship['theta'][t_ref])
            x_ref = float(reference_ship['x'][t_ref])
            y_ref = float(reference_ship['y'][t_ref])
            cos_t = np.cos(theta_ref)
            sin_t = np.sin(theta_ref)
            
            for i, ship in enumerate(processed_ships):
                feats_g = all_features_global[i]
                
                # 转换到统一的局部坐标系（相对于参考船）
                dx = feats_g['x'] - x_ref
                dy = feats_g['y'] - y_ref
                x_local =  cos_t * dx + sin_t * dy
                y_local = -sin_t * dx + cos_t * dy
                
                vx_local =  cos_t * feats_g['vx'] + sin_t * feats_g['vy']
                vy_local = -sin_t * feats_g['vx'] + cos_t * feats_g['vy']
                
                ax_local =  cos_t * feats_g['ax'] + sin_t * feats_g['ay']
                ay_local = -sin_t * feats_g['ax'] + cos_t * feats_g['ay']
                
                theta_local = self.wrap_angle_rad(feats_g['theta'] - theta_ref)
                omega_local = feats_g['omega']
                
                features_local = {
                    'time_seconds': feats_g['time_seconds'],
                    'x': x_local,
                    'y': y_local,
                    'vx': vx_local,
                    'vy': vy_local,
                    'ax': ax_local,
                    'ay': ay_local,
                    'v_magnitude': feats_g.get('v_magnitude', np.sqrt(vx_local ** 2 + vy_local ** 2)),
                    'a_magnitude': feats_g.get('a_magnitude', np.sqrt(ax_local ** 2 + ay_local ** 2)),
                    'theta': theta_local,
                    'theta_deg': np.degrees(theta_local),
                    'omega': omega_local,
                    'mmsi': feats_g['mmsi']
                }
                
                ship['features_local'] = features_local
        
        # 统计信息
        time_steps = len(processed_ships[0]['features_local']['time_seconds'])
        
        # 构建结果
        result = {
            'metadata': {
                'ship_count': len(trajectories),
                'time_steps': time_steps,
                'time_step_size': self.target_time_step,
                'origin_lon': origin_lon,
                'origin_lat': origin_lat,
                'origin_time': origin_time,
                'total_duration': (time_steps - 1) * self.target_time_step
            },
            'ships': processed_ships,
            'impacts': impacts,
            'original_impacts_shape': impacts.shape
        }
        
        if self.verbose:
            print(f"预处理完成：{len(processed_ships)}艘船只，{time_steps}个时间步")
        
        return result
    
    def print_scene_summary(self, processed_scene: Dict[str, Any]):
        """打印预处理后场景的摘要信息"""
        metadata = processed_scene['metadata']
        ships = processed_scene['ships']
        
        print(f"\n=== 预处理场景摘要 ===")
        print(f"船只数量: {metadata['ship_count']}")
        print(f"时间步数: {metadata['time_steps']}")
        print(f"时间步长: {metadata['time_step_size']}秒")
        print(f"总持续时间: {metadata['total_duration']}秒")
        print(f"原点坐标: ({metadata['origin_lon']:.6f}, {metadata['origin_lat']:.6f})")
        
        print(f"\n前3艘船只的局部特征范围(相对T_h-1):")
        for i, ship in enumerate(ships[:3]):
            features = ship['features_local']
            print(f"  船只{i+1} (MMSI: {ship['mmsi']}):")
            print(f"    X范围: {features['x'].min():.1f} ~ {features['x'].max():.1f} m")
            print(f"    Y范围: {features['y'].min():.1f} ~ {features['y'].max():.1f} m") 
            print(f"    速度范围: {features['v_magnitude'].min():.2f} ~ {features['v_magnitude'].max():.2f} m/s")
            print(f"    加速度范围: {features['a_magnitude'].min():.3f} ~ {features['a_magnitude'].max():.3f} m/s²")
            print(f"    航向变化率范围: {features['omega'].min():.4f} ~ {features['omega'].max():.4f} rad/s")

def main():
    """测试函数"""
    import pickle
    
    # 加载测试数据
    data_path = '/home/mahexing/ais_data_process/scene_generation/DI-MTP/data/per_file/POS_OK_2024-07-01_Waigaoqiao_Port_processed_batches.pkl'
    
    print("加载测试数据...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 创建预处理器（使用30秒时间步长）
    preprocessor = MaritimeScenePreprocessor(target_time_step=30.0, num_historical_steps=5)
    
    # 处理第一个场景
    scene = data[0]
    processed = preprocessor.preprocess_scene(scene)
    
    # 显示结果摘要
    preprocessor.print_scene_summary(processed)
    
    # 创建HeteroData结构
    print(f"\n=== 创建PyTorch Geometric HeteroData结构 ===")
    hetero_data = preprocessor.create_hetero_data(processed)
    
    # 显示HeteroData详细结构
    preprocessor.print_hetero_data_info(hetero_data)
    
    # 验证HeteroData结构
    print(f"\n=== HeteroData结构验证 ===")
    validation_results = preprocessor.validate_hetero_data(hetero_data)
    
    if validation_results['structure_valid']:
        print("✅ HeteroData结构验证通过!")
    else:
        print("❌ HeteroData结构验证失败!")
        for error in validation_results['errors']:
            print(f"  错误: {error}")
    
    # 显示HeteroData统计信息
    stats = validation_results['statistics']
    print(f"\n📊 HeteroData统计信息:")
    print(f"   船只数量: {stats.get('num_ships', 'N/A')}")
    print(f"   时间步数: {stats.get('num_timesteps', 'N/A')}")
    print(f"   特征维度: {stats.get('num_features', 'N/A')}")
    print(f"   特征张量形状: {stats.get('agent_features_shape', 'N/A')}")
    print(f"   特征数据类型: {stats.get('agent_features_dtype', 'N/A')}")
    
    # 边关系统计
    if 'edge_types' in stats:
        print(f"\n🔗 图边关系统计:")
        for edge_info in stats['edge_types']:
            print(f"   {edge_info['type']}: {edge_info['num_edges']} 条边, 形状: {edge_info['shape']}")
    
    # 特征归一化统计
    if 'feature_stats' in stats:
        feature_stats = stats['feature_stats']
        print(f"\n📈 归一化后特征统计:")
        feature_names = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'theta', 'omega']
        print(f"   特征均值: {[f'{feature_names[i]}={val:.3f}' for i, val in enumerate(feature_stats['mean'])]}")
        print(f"   特征标准差: {[f'{feature_names[i]}={val:.3f}' for i, val in enumerate(feature_stats['std'])]}")
        print(f"   全局最小值: {feature_stats['min']:.3f}")
        print(f"   全局最大值: {feature_stats['max']:.3f}")
    
    # 显示归一化统计信息
    if hasattr(hetero_data, 'normalization_stats'):
        print(f"\n🔄 归一化统计信息 (原始数据):")
        norm_stats = hetero_data.normalization_stats
        for feature, stats in norm_stats.items():
            print(f"   {feature}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
    
    # 显示第一艘船的详细局部特征
    print(f"\n=== 第一艘船详细局部特征 (前3个时间步, 相对T_h-1) ===")
    first_ship = processed['ships'][0]['features_local']
    for i in range(min(3, len(first_ship['time_seconds']))):
        print(f"时间步 {i}: t={first_ship['time_seconds'][i]:.1f}s")
        print(f"  位置: ({first_ship['x'][i]:.1f}, {first_ship['y'][i]:.1f}) m")
        print(f"  速度: ({first_ship['vx'][i]:.2f}, {first_ship['vy'][i]:.2f}) m/s, |v|={first_ship['v_magnitude'][i]:.2f} m/s")
        print(f"  加速度: ({first_ship['ax'][i]:.3f}, {first_ship['ay'][i]:.3f}) m/s²")
        print(f"  航向: {first_ship['theta_deg'][i]:.1f}°, 角速度: {first_ship['omega'][i]:.4f} rad/s")
        print()
    
    print(f"=== 预处理和验证完成 ===")
    return hetero_data, validation_results

if __name__ == "__main__":
    main()
