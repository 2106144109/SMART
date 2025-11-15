#!/usr/bin/env python3
"""
Maritime Trajectory Clustering Script
为海上场景创建轨迹词汇表（Trajectory Vocabulary）
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/mahexing/SMART-main')

from smart.utils.geometry import wrap_angle


def wrap_angle_np(angle, min_val=-np.pi, max_val=np.pi):
    """将角度包裹到[min_val, max_val)（NumPy 版）。"""
    return min_val + (angle + max_val) % (max_val - min_val)


def normalize_trajectories(trajectories):
    """
    将轨迹归一化到相对起点/起始朝向的坐标系。

    Args:
        trajectories: [N, shift+1, 3]，(x, y, theta)

    Returns:
        traj_norm: [N, shift+1, 3]
    """
    traj = trajectories.copy()
    # 1) 相对起点平移
    traj[:, :, 0:2] = traj[:, :, 0:2] - traj[:, 0:1, 0:2]
    # 2) 旋转到首帧朝向的局部坐标（绕 -theta0）
    theta0 = traj[:, 0:1, 2]
    cos0 = np.cos(theta0)
    sin0 = np.sin(theta0)
    x = traj[:, :, 0]
    y = traj[:, :, 1]
    x_r =  cos0 * x + sin0 * y
    y_r = -sin0 * x + cos0 * y
    traj[:, :, 0] = x_r
    traj[:, :, 1] = y_r
    # 3) 航向相对化并做角度包裹
    traj[:, :, 2] = wrap_angle_np(traj[:, :, 2] - theta0)
    return traj


def average_distance_vectorized(point_set1, centroids):
    """计算轨迹到聚类中心的平均距离（向量化）"""
    dists = np.sqrt(np.sum((point_set1[:, None, :, :] - centroids[None, :, :, :])**2, axis=-1))
    return np.mean(dists, axis=2)


def assign_clusters(sub_X, centroids):
    """将轨迹分配到最近的聚类中心"""
    distances = average_distance_vectorized(sub_X, centroids)
    return np.argmin(distances, axis=1)


def Kdisk_cluster(X, N=256, tol=0.1, width=0, length=0, a_pos=None, 
                  x_min=-50, x_max=100, y_min=-50, y_max=50, cal_mean_heading=True):
    """
    K-disk 聚类算法（适配海上场景）
    
    Args:
        X: 轨迹多边形轮廓 [num_traj, 4, 2]
        N: 目标聚类数量
        tol: 容差（控制聚类紧密程度）
        width: 船舶宽度
        length: 船舶长度
        a_pos: 原始轨迹数据 [num_traj, time_steps, 3] (x, y, theta)
        x_min, x_max, y_min, y_max: 有效区域边界
        cal_mean_heading: 是否计算平均航向
    
    Returns:
        centroids: 聚类中心 [N, 4, 2]
        ret_traj: 代表性轨迹 [N, time_steps, 3]
    """
    S = []
    ret_traj_list = []
    iteration = 0
    max_iterations = N * 10  # 防止无限循环
    
    print(f"开始聚类: 目标{N}个簇, 容差={tol:.3f}, 船舶尺寸={width:.1f}x{length:.1f}m")
    
    while len(S) < N and iteration < max_iterations:
        iteration += 1
        num_all = X.shape[0]
        
        if num_all == 0:
            print(f"⚠️ 警告: 剩余轨迹数为0，已获得{len(S)}个聚类中心")
            break
        
        # 随机选择一个轨迹作为候选中心
        choice_index = np.random.choice(num_all)
        x0 = X[choice_index]
        
        # 边界检查：确保中心点在合理范围内
        center_x = x0[0, 0]
        center_y = x0[0, 1]
        if center_x < x_min or center_x > x_max or center_y < y_min or center_y > y_max:
            continue
        
        # 计算距离并分类
        distances = np.sum((X - x0)**2, axis=(1, 2)) / 4
        res_mask = distances > (tol**2)  # 保留的轨迹
        del_mask = distances <= (tol**2)  # 删除的轨迹（属于当前簇）
        
        if cal_mean_heading and del_mask.sum() > 0:
            # 计算簇内轨迹的平均航向
            del_contour = X[del_mask]
            diff_xy = del_contour[:, 0, :] - del_contour[:, 3, :]
            del_heading = np.arctan2(diff_xy[:, 1], diff_xy[:, 0]).mean()
            
            # 使用平均航向重新计算多边形轮廓
            x0_center_x = x0.mean(0)[0]
            x0_center_y = x0.mean(0)[1]
            x0 = cal_polygon_contour(x0_center_x, x0_center_y, del_heading, width, length)
            
            # 使用簇内轨迹的平均轨迹作为代表
            del_traj = a_pos[del_mask]
            ret_traj = del_traj.mean(0)[None, ...]
        else:
            x0 = x0[None, ...]
            ret_traj = a_pos[choice_index][None, ...]
        
        # 更新数据集（移除已聚类的轨迹）
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        
        S.append(x0)
        ret_traj_list.append(ret_traj)
        
        if len(S) % 50 == 0:
            print(f"  进度: {len(S)}/{N} 个聚类中心, 剩余轨迹: {X.shape[0]}")
    
    if len(S) < N:
        print(f"⚠️ 警告: 仅获得{len(S)}/{N}个聚类中心（可能需要减小tol或增加数据量）")
    
    centroids = np.concatenate(S, axis=0)
    ret_traj = np.concatenate(ret_traj_list, axis=0)
    
    return centroids, ret_traj


def cal_polygon_contour(x, y, theta, width, length):
    """
    计算矩形船舶的四个角点坐标
    
    Args:
        x, y: 船舶中心坐标
        theta: 航向角（弧度）
        width: 船宽
        length: 船长
    
    Returns:
        polygon_contour: [4, 2] 四个角点 [左前, 右前, 右后, 左后]
    """
    # 左前角
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))
    
    # 右前角
    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))
    
    # 右后角
    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))
    
    # 左后角
    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))
    
    polygon_contour = np.concatenate((
        left_front[:, None, :], 
        right_front[:, None, :], 
        right_back[:, None, :], 
        left_back[:, None, :]
    ), axis=1)
    
    return polygon_contour


def load_maritime_trajectories(data_dirs, max_samples=100000, shift=1):
    """
    从maritime数据集中加载轨迹数据
    
    Args:
        data_dirs: 数据目录列表
        max_samples: 最大加载样本数
        shift: 窗口跨度（用于运动token，1 表示提取两帧 [t, t+1]）
    
    Returns:
        trajectories: [N, shift+1, 3] (x, y, theta)
    """
    print("\n" + "=" * 70)
    print("📊 从Maritime数据集加载轨迹")
    print("=" * 70)
    
    all_trajectories = []
    sample_count = 0
    
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"⚠️ 目录不存在: {data_dir}")
            continue
        
        # 获取所有.pt文件
        pt_files = sorted(list(data_dir.glob("*.pt")))
        print(f"\n目录: {data_dir}")
        print(f"找到 {len(pt_files)} 个文件")
        
        for pt_file in tqdm(pt_files, desc="加载文件"):
            try:
                # 加载文件
                data_list = torch.load(pt_file, map_location='cpu', weights_only=False)
                
                if not isinstance(data_list, list):
                    data_list = [data_list]
                
                # 遍历文件中的每个场景
                for scene_data in data_list:
                    if sample_count >= max_samples:
                        break
                    
                    # 验证数据格式
                    if not hasattr(scene_data, 'node_types') or 'agent' not in scene_data.node_types:
                        continue
                    
                    # 获取agent节点数据
                    agent_data = scene_data['agent']
                    
                    # x: [N_ships, T_steps, F_features]
                    # 我们需要: x (0), y (1), theta (6)
                    if not hasattr(agent_data, 'x'):
                        continue
                    
                    features = agent_data.x  # [N_ships, T_steps, F_features]
                    
                    # 检查形状
                    if features.dim() != 3 or features.shape[2] < 7:
                        continue
                    
                    N_ships = features.shape[0]
                    T_steps = features.shape[1]
                    
                    # 提取每艘船的短程2步轨迹（滑动窗口）
                    for ship_idx in range(N_ships):
                        # 检查是否有足够的时间步
                        if T_steps < shift + 1:
                            continue
                        
                        # 遍历所有起点 t，收集 [t, t+shift] 两帧片段
                        for t in range(0, T_steps - shift):
                            end_idx = t + shift + 1
                            # 可选：若存在 valid_mask，要求窗口内均有效
                            vmask_ok = True
                            if hasattr(agent_data, 'valid_mask'):
                                vmask_seg = agent_data.valid_mask[ship_idx, t:end_idx].cpu().numpy()
                                vmask_ok = bool(vmask_seg.all())
                            if not vmask_ok:
                                continue
                            
                            # 用速度积分重建相对位移（速度更可靠）
                            # 特征索引: [x, y, vx, vy, ax, ay, theta, omega]
                            vx = features[ship_idx, t:end_idx, 2].numpy()
                            vy = features[ship_idx, t:end_idx, 3].numpy()
                            theta = features[ship_idx, t:end_idx, 6].numpy()
                            
                            x = np.zeros(shift + 1)
                            y = np.zeros(shift + 1)
                            x[0] = 0.0
                            y[0] = 0.0
                            dt = 30.0
                            for i in range(1, shift + 1):
                                x[i] = x[i-1] + vx[i-1] * dt
                                y[i] = y[i-1] + vy[i-1] * dt
                            
                            # 略过全零片段
                            if np.all(x == 0) and np.all(y == 0):
                                continue
                            
                            trajectory = np.stack([x, y, theta], axis=1)  # [shift+1, 3]
                            all_trajectories.append(trajectory)
                            sample_count += 1
                            if sample_count >= max_samples:
                                break
                        if sample_count >= max_samples:
                            break
                    
                    if sample_count >= max_samples:
                        break
                        
            except Exception as e:
                print(f"\n⚠️ 加载文件失败 {pt_file.name}: {e}")
                continue
            
            if sample_count >= max_samples:
                break
        
        if sample_count >= max_samples:
            break
    
    if len(all_trajectories) == 0:
        raise ValueError("未能加载任何轨迹数据！")
    
    trajectories = np.stack(all_trajectories, axis=0)  # [N, shift+1, 3]
    
    print(f"\n✅ 成功加载 {len(trajectories)} 条轨迹")
    print(f"   形状: {trajectories.shape}")
    print(f"   X范围: [{trajectories[:, :, 0].min():.2f}, {trajectories[:, :, 0].max():.2f}]")
    print(f"   Y范围: [{trajectories[:, :, 1].min():.2f}, {trajectories[:, :, 1].max():.2f}]")
    print(f"   Theta范围: [{trajectories[:, :, 2].min():.2f}, {trajectories[:, :, 2].max():.2f}]")
    
    return trajectories


def create_maritime_vocabulary(data_dirs, 
                               num_clusters=256,
                               shift=1,
                               max_samples=100000,
                               ship_width=10.0,
                               ship_length=50.0,
                               tolerance=0.15,
                               w_ang=100.0,
                               output_path='data/maritime_motion_vocab.pt'):
    """
    为海上场景创建轨迹词汇表
    
    Args:
        data_dirs: 数据目录列表
        num_clusters: 词汇表大小（聚类数量）
        shift: 窗口跨度（生成 shift+1 个时间点；1 表示两帧片段）
        max_samples: 最大使用样本数
        ship_width: 船舶宽度（米）
        ship_length: 船舶长度（米）
        tolerance: 聚类容差
        output_path: 输出文件路径
    """
    print("\n" + "=" * 70)
    print("🚢 Maritime轨迹聚类 - 创建运动词汇表")
    print("=" * 70)
    print(f"参数配置:")
    print(f"  词汇表大小: {num_clusters}")
    print(f"  时间步数: {shift + 1}")
    print(f"  船舶尺寸: {ship_width}m x {ship_length}m")
    print(f"  聚类容差: {tolerance}")
    print(f"  最大样本数: {max_samples}")
    print(f"  角度权重 w_ang: {w_ang:.2f} (米^2)")
    
    # 1. 加载轨迹数据
    trajectories = load_maritime_trajectories(data_dirs, max_samples, shift)
    
    # 2. 随机采样（如果数据太多）
    if trajectories.shape[0] > max_samples:
        print(f"\n📉 降采样: {trajectories.shape[0]} -> {max_samples}")
        indices = np.random.choice(trajectories.shape[0], max_samples, replace=False)
        trajectories = trajectories[indices]
    
    # 3. 归一化到相对起点/起始朝向（含旋转对齐）
    print("\n🔄 归一化短程片段到相对坐标系（起点/朝向对齐）...")
    traj_norm = normalize_trajectories(trajectories)  # [N, shift+1, 3]

    # 4. 展平并执行 KMeans（2步短程片段）
    print("\n🎯 执行 KMeans 聚类（2步短程片段）...")
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        print("❌ 需要 scikit-learn：pip install scikit-learn")
        raise

    # 4.1 用 (x, y, sqrt(w_ang)*cosθ, sqrt(w_ang)*sinθ) 作为聚类特征（统一度量）
    print("\n⚙️  特征转换: 使用 (x, y, sqrt(w_ang)*cosθ, sqrt(w_ang)*sinθ) ...")
    N = traj_norm.shape[0]
    S = shift + 1
    x = traj_norm[..., 0]
    y = traj_norm[..., 1]
    theta = traj_norm[..., 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sqrt_w = float(np.sqrt(w_ang))
    features_transformed = np.stack([x, y, sqrt_w * cos_theta, sqrt_w * sin_theta], axis=-1)  # [N, S, 4]
    data_for_cluster = features_transformed.reshape(N, S * 4).astype(np.float32)

    n_clusters_eff = min(num_clusters, data_for_cluster.shape[0])
    if n_clusters_eff <= 0:
        raise ValueError("有效聚类数为0，请检查数据量或参数设置")

    kmeans = KMeans(n_clusters=n_clusters_eff, n_init=10, random_state=0)
    kmeans.fit(data_for_cluster)
    centers = kmeans.cluster_centers_.astype(np.float32)    # [K, S*4]
    centers_reshaped = centers.reshape(n_clusters_eff, S, 4)
    x_c = centers_reshaped[..., 0]
    y_c = centers_reshaped[..., 1]
    cos_c_w = centers_reshaped[..., 2]
    sin_c_w = centers_reshaped[..., 3]
    # 还原未加权的 cos/sin，用于计算角度
    cos_c = cos_c_w / sqrt_w
    sin_c = sin_c_w / sqrt_w
    theta_c = np.arctan2(sin_c, cos_c)
    ret_traj = np.stack([x_c, y_c, theta_c], axis=-1).astype(np.float32)  # [K, S, 3]
    # 角度规范到 (-pi, pi]
    ret_traj[:, :, 2] = wrap_angle_np(ret_traj[:, :, 2])

    # 以代表性轨迹的末时刻计算多边形作为 token（与原结构一致）
    centroids = cal_polygon_contour(
        ret_traj[:, -1, 0],
        ret_traj[:, -1, 1],
        ret_traj[:, -1, 2],
        ship_width,
        ship_length
    )
    
    print(f"\n✅ 聚类完成!")
    print(f"   获得 {centroids.shape[0]} 个聚类中心")
    print(f"   代表性轨迹形状: {ret_traj.shape}")
    
    # 6. 重新计算完整时间序列的多边形
    print("\n📊 计算完整时间序列的多边形...")
    num_actual_clusters = ret_traj.shape[0]
    contour_all = cal_polygon_contour(
        ret_traj[:, :, 0].reshape(num_actual_clusters * (shift + 1)),
        ret_traj[:, :, 1].reshape(num_actual_clusters * (shift + 1)),
        ret_traj[:, :, 2].reshape(num_actual_clusters * (shift + 1)),
        ship_width,
        ship_length
    )
    contour_all = contour_all.reshape(num_actual_clusters, (shift + 1), 4, 2)
    
    # 7. 保存词汇表
    vocab = {
        'token': {'ship': centroids},  # [N, 4, 2] 聚类中心的多边形
        'traj': {'ship': ret_traj},    # [N, shift+1, 3] 代表性轨迹
        'token_all': {'ship': contour_all},  # [N, shift+1, 4, 2] 完整时间序列
        'metadata': {
            'num_clusters': num_actual_clusters,
            'shift': shift,
            'ship_width': ship_width,
            'ship_length': ship_length,
            'tolerance': tolerance,
            'num_samples': trajectories.shape[0],
            'method': 'kmeans_xy_cossin_weighted',
            'w_ang': float(w_ang)
        }
    }
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    torch.save(vocab, output_path)
    print(f"\n💾 词汇表已保存: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 8. 打印统计信息
    print("\n" + "=" * 70)
    print("📈 词汇表统计")
    print("=" * 70)
    print(f"词汇表大小: {num_actual_clusters}")
    print(f"时间步数: {shift + 1}")
    print(f"船舶尺寸: {ship_width}m x {ship_length}m")
    print(f"\n轨迹统计:")
    print(f"  X位移范围: [{ret_traj[:, :, 0].min():.2f}, {ret_traj[:, :, 0].max():.2f}]")
    print(f"  Y位移范围: [{ret_traj[:, :, 1].min():.2f}, {ret_traj[:, :, 1].max():.2f}]")
    print(f"  航向范围: [{ret_traj[:, :, 2].min():.2f}, {ret_traj[:, :, 2].max():.2f}]")
    
    return vocab


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Maritime轨迹聚类 - 创建Token词表')
    parser.add_argument('--data_dirs', type=str, nargs='+', 
                        default=['data/maritime_windows_30s_no_norm/train'],
                        help='训练数据目录列表')
    parser.add_argument('--output', type=str, 
                        default='smart/tokens/maritime_tokens_no_norm.pkl',
                        help='输出Token词表路径')
    parser.add_argument('--token_size', type=int, default=2048,
                        help='Token词表大小（聚类数量）')
    parser.add_argument('--shift', type=int, default=1,
                        help='窗口跨度（生成 shift+1 个时间点；1 表示两帧片段）')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='最大使用样本数')
    parser.add_argument('--ship_width', type=float, default=10.0,
                        help='船舶宽度（米）')
    parser.add_argument('--ship_length', type=float, default=50.0,
                        help='船舶长度（米）')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='聚类容差（越小越紧密，建议0.08-0.15）')
    parser.add_argument('--w_ang', type=float, default=100.0,
                        help='角度权重（米^2），确定 cos/sin 对距离的贡献')
    
    args = parser.parse_args()
    
    # 配置参数
    config = {
        'data_dirs': args.data_dirs,
        'num_clusters': args.token_size,
        'shift': args.shift,
        'max_samples': args.max_samples,
        'ship_width': args.ship_width,
        'ship_length': args.ship_length,
        'tolerance': args.tolerance,
        'w_ang': args.w_ang,
        'output_path': args.output
    }
    
    print("\n" + "=" * 70)
    print("🚢 Maritime轨迹聚类脚本")
    print("=" * 70)
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建词汇表
    vocab = create_maritime_vocabulary(**config)
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print(f"\n下一步:")
    print(f"1. 检查生成的词汇表: {config['output_path']}")
    print(f"2. 验证Token方向分布（应该均衡）")
    print(f"3. 更新训练配置文件")
    print(f"4. 开始训练模型")

