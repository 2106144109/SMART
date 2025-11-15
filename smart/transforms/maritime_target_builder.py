#!/usr/bin/env python3
"""
Maritime Target Builder
海上场景的目标构建器
"""

import torch
import numpy as np
import pickle
import os
from torch_geometric.transforms import BaseTransform
from smart.utils import wrap_angle


class MaritimeTargetBuilder(BaseTransform):
    """
    海上场景的目标构建器
    由于数据已经在预处理阶段准备好，这里只做基本的验证和可选的增强
    """
    
    def __init__(self,
                 num_historical_steps: int = 5,
                 num_future_steps: int = 16,
                 mode: str = "train",
                 token_size: int = 2048) -> None:
        """
        Args:
            num_historical_steps: 历史步数（默认5步，30秒间隔）
            num_future_steps: 未来预测步数（默认16步，30秒间隔）
            mode: 'train', 'val', or 'test'
        """
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.training = mode == "train"
        self.shift = 1  # 30秒间隔，shift=1
        self.token_size = token_size
        
        # 加载maritime token库
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        token_path = os.path.join(module_dir, 'smart/tokens/maritime_tokens_no_norm.pkl')

        # 统一成字典形式：优先期望 {'ship': ndarray, ...}
        self.trajectory_token = None
        if os.path.exists(token_path):
            # 兼容两种保存方式：优先 torch.load，失败回退 pickle.load
            try:
                token_data = torch.load(token_path, map_location='cpu', weights_only=False)
            except Exception:
                with open(token_path, 'rb') as f:
                    token_data = pickle.load(f)
            if isinstance(token_data, dict) and ('traj' in token_data):
                traj_val = token_data['traj']
                if isinstance(traj_val, dict):
                    self.trajectory_token = traj_val
                else:
                    # 若为 ndarray，包一层，作为 ship 类使用
                    self.trajectory_token = {'ship': traj_val}
                # 读取加权度量超参（离线保存），默认100.0
                meta = token_data.get('metadata', {}) if isinstance(token_data, dict) else {}
                self.w_ang = float(meta.get('w_ang', 100.0))
                self._sqrt_w_ang = float(np.sqrt(self.w_ang))
                # 选择库键并预计算库的加权特征 [K, S*4]
                if isinstance(self.trajectory_token, dict) and len(self.trajectory_token) > 0:
                    self._lib_key = 'ship' if ('ship' in self.trajectory_token) else next(iter(self.trajectory_token.keys()))
                    lib_np = self.trajectory_token[self._lib_key]  # [K, S, 3]
                    lib = torch.from_numpy(lib_np).float()
                    cos_c = torch.cos(lib[..., 2])
                    sin_c = torch.sin(lib[..., 2])
                    # (x, y, sqrt(w_ang)*cos, sqrt(w_ang)*sin)
                    lib_feats = torch.stack([lib[..., 0], lib[..., 1], cos_c * self._sqrt_w_ang, sin_c * self._sqrt_w_ang], dim=-1)
                    self._lib_feats_w = lib_feats.reshape(lib_feats.shape[0], -1).contiguous()  # [K, S*4]
                    # 校验词表大小与配置是否一致
                    num_tokens = self._lib_feats_w.shape[0]
                    if isinstance(self.token_size, int) and self.token_size > 0 and num_tokens != self.token_size:
                        print(f"[MaritimeTargetBuilder] 警告: 词表大小({num_tokens})与配置token_size({self.token_size})不一致")
                else:
                    self._lib_key = None
                    self._lib_feats_w = torch.zeros((0, (self.shift + 1) * 4), dtype=torch.float32)
            else:
                self.trajectory_token = None
    
    def __call__(self, data):
        """
        处理数据样本
        
        Args:
            data: HeteroData对象
            
        Returns:
            处理后的HeteroData对象
        """
        # 数据已经在预处理阶段完全准备好了
        # 这里只做基本验证和可选的数据增强
        
        # 更详细的检查
        if not hasattr(data, 'node_types'):
            raise ValueError(f"数据不是HeteroData对象，类型: {type(data)}")
        
        if 'agent' not in data.node_types:
            actual_nodes = list(data.node_types) if hasattr(data, 'node_types') else []
            raise ValueError(f"数据中缺少 'agent' 节点。实际节点: {actual_nodes}")
        
        agent = data['agent']
        
        # 验证数据形状
        if hasattr(agent, 'x'):
            num_ships, total_steps, num_features = agent.x.shape
            
            # 检查时间步数是否匹配
            expected_steps = self.num_historical_steps + self.num_future_steps
            if total_steps != expected_steps:
                # 如果不匹配，尝试裁剪或填充
                if total_steps > expected_steps:
                    # 裁剪到期望长度
                    agent.x = agent.x[:, :expected_steps, :]
                    if hasattr(agent, 'valid_mask'):
                        agent.valid_mask = agent.valid_mask[:, :expected_steps]
                elif total_steps < expected_steps:
                    # 数据太短，报错
                    raise ValueError(
                        f"数据时间步数不足: {total_steps} < {expected_steps}"
                    )
        
        # 确保有必要的属性
        if not hasattr(agent, 'num_nodes'):
            agent.num_nodes = agent.x.shape[0]
        
        if not hasattr(agent, 'av_index'):
            # 默认第一艘船为关注对象
            agent.av_index = 0
        
        # ========== 添加Token化所需的字段 ==========
        # 模型需要这些字段来进行token预测
        
        # 从特征中提取位置和航向
        # features: [N_ships, T_steps, 8] 其中8维=[x, y, vx, vy, ax, ay, theta, omega]
        num_ships = agent.x.shape[0]
        num_steps = agent.x.shape[1]
        
        # 1. token_pos: 使用完整的8维特征 [N_ships, T_steps, 8]
        # 模型的input_dim=8，需要完整特征向量
        agent['token_pos'] = agent.x  # 使用完整的8维特征
        
        # 2. token_heading: 使用theta航向 [N_ships, T_steps]
        agent['token_heading'] = agent.x[:, :, 6]  # 取theta
        
        # 3. agent_valid_mask: 与valid_mask相同
        if hasattr(agent, 'valid_mask'):
            agent['agent_valid_mask'] = agent.valid_mask
        else:
            # 如果没有valid_mask，创建一个全True的mask
            agent['agent_valid_mask'] = torch.ones(num_ships, num_steps, dtype=torch.bool)
        
        # 4. type: Agent类型 [N_ships]
        # Maritime场景：所有都是船舶(ship)，用type=0表示（对应Waymo的vehicle）
        if not hasattr(agent, 'type'):
            agent['type'] = torch.zeros(num_ships, dtype=torch.uint8)  # 0=ship/vehicle
        
        # 5. category: Agent类别 [N_ships]
        # 用于标记重要性，3表示需要预测的agent
        if not hasattr(agent, 'category'):
            agent['category'] = torch.ones(num_ships, dtype=torch.uint8) * 3  # 3=需要预测
        
        # 6. token_idx: 匹配轨迹到token库的索引
        # 计算token化后的步数
        num_token_steps = (num_steps - 1) // self.shift  # 滑动窗口token化
        
        if isinstance(self.trajectory_token, dict) and (len(self.trajectory_token) > 0):
            # 匹配轨迹到最近的token（优先 ship，否则回退其他键）
            token_idx = self._match_tokens_to_library(agent, num_ships, num_steps)
        else:
            # 如果没有token库，创建随机索引作为占位符
            # 注意：必须使用num_steps而不是num_token_steps，以保持与_match_tokens_to_library一致
            token_idx = torch.randint(0, int(self.token_size), (num_ships, num_steps), dtype=torch.int64)
        
        agent['token_idx'] = token_idx
        
        # 7. token_velocity: 从token_pos计算速度 [N_ships, T_steps, 8]
        # 时间步长为30秒
        dt = 30.0  # 30秒时间步
        
        # 计算速度：(pos[t] - pos[t-1]) / dt
        # 第一步速度为0，后续步从差分计算
        token_velocity = torch.cat([
            agent['token_pos'].new_zeros(num_ships, 1, 8),  # 第一步速度为0
            (agent['token_pos'][:, 1:, :] - agent['token_pos'][:, :-1, :]) / dt  # 差分/dt
        ], dim=1)  # [N_ships, T_steps, 8]

        # 对 theta 采用环绕归一化的差分，避免在 ±pi 处的跳变
        delta_theta = wrap_angle(agent['token_pos'][:, 1:, 6] - agent['token_pos'][:, :-1, 6])
        token_velocity[:, 1:, 6] = delta_theta / dt
        
        agent['token_velocity'] = token_velocity
        
        # 8. shape: Agent的形状信息 [N_ships, T_steps, 3] (length, width, height)
        # Maritime场景：使用固定的船舶尺寸
        ship_length = 50.0  # 米
        ship_width = 10.0   # 米
        ship_height = 5.0   # 米（估计值）
        
        # 使用repeat而不是expand，避免batch合并问题
        agent['shape'] = torch.tensor(
            [ship_length, ship_width, ship_height], 
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).repeat(num_ships, num_steps, 1)  # [N_ships, T_steps, 3]
        
        # 9. 添加空的pt_token节点（Maritime场景无地图）
        # 模型代码会访问pt_token节点，需要存在但为空
        if 'pt_token' not in data.node_types:
            # 创建空的pt_token节点
            data['pt_token'].num_nodes = 0
            data['pt_token']['position'] = torch.zeros((0, 3), dtype=torch.float32)
            data['pt_token']['orientation'] = torch.zeros((0,), dtype=torch.float32)
            # 不添加batch字段，会在Batch.from_data_list时自动添加
        
        # 可选：训练时的数据增强
        if self.training:
            # 这里可以添加数据增强逻辑，例如：
            # - 随机旋转
            # - 添加噪声
            # - 随机丢弃某些船只
            pass
        
        return data
    
    def _match_tokens_to_library(self, agent, num_ships, num_steps):
        """
        将实际轨迹匹配到token库
        
        Returns:
            token_idx: [N_ships, num_token_steps] 每个时间窗口匹配的token索引
        """
        # 选择 token 库：优先 'ship'，否则使用首个可用键
        lib = self.trajectory_token
        key = 'ship' if ('ship' in lib) else (next(iter(lib.keys())) if isinstance(lib, dict) and len(lib) > 0 else None)
        if key is None:
            token_library = torch.zeros((512, 2, 3), dtype=torch.float32)
        else:
            token_library = torch.from_numpy(lib[key]).float()  # [K, 2, 3]
        num_tokens = token_library.shape[0]
        
        # 提取位置和航向
        pos = agent.x[:, :, :2]  # [N_ships, T_steps, 2]
        theta = agent.x[:, :, 6]  # [N_ships, T_steps]
        
        # 使用库的窗口长度以保证与离线一致，并一次性提取窗口
        S_lib = (self._lib_feats_w.shape[1] // 4) if hasattr(self, '_lib_feats_w') and self._lib_feats_w.numel() > 0 else (self.shift + 1)
        start_t = self.num_historical_steps
        end_exclusive = num_steps - (S_lib - 1)
        window_starts = list(range(start_t, end_exclusive, self.shift))
        num_windows = len(window_starts)

        if num_windows > 0:
            # 堆叠所有窗口 [N, W, S, 2] / [N, W, S]
            pos_windows = torch.stack([pos[:, t:t+S_lib, :] for t in window_starts], dim=1)
            theta_windows = torch.stack([theta[:, t:t+S_lib] for t in window_starts], dim=1)

            # 相对平移到首帧
            pos_rel = pos_windows - pos_windows[:, :, :1, :]
            theta0 = theta_windows[:, :, 0]

            # 旋转对齐到首帧朝向
            cos0 = torch.cos(theta0).unsqueeze(-1)
            sin0 = torch.sin(theta0).unsqueeze(-1)
            x = pos_rel[..., 0]
            y = pos_rel[..., 1]
            x_r =  cos0 * x + sin0 * y
            y_r = -sin0 * x + cos0 * y
            pos_rel_rot = torch.stack([x_r, y_r], dim=-1)  # [N, W, S, 2]

            # 相对航向并做环绕
            theta_rel = wrap_angle(theta_windows - theta0.unsqueeze(-1))  # [N, W, S]

            # 组合特征并加权: (x, y, sqrt(w_ang)*cosθ, sqrt(w_ang)*sinθ)
            cos_r = torch.cos(theta_rel)
            sin_r = torch.sin(theta_rel)
            win_feats = torch.stack([
                pos_rel_rot[..., 0],
                pos_rel_rot[..., 1],
                cos_r * self._sqrt_w_ang,
                sin_r * self._sqrt_w_ang
            ], dim=-1)  # [N, W, S, 4]

            # 展平为 [N*W, S*4]
            N_ships_eff = win_feats.shape[0]
            win_flat = win_feats.reshape(N_ships_eff * num_windows, -1)

            lib_feats = self._lib_feats_w.to(win_flat.device)
            if lib_feats.numel() == 0:
                token_idx_windows = torch.zeros((N_ships_eff, num_windows), dtype=torch.int64, device=win_flat.device)
            else:
                dist = torch.cdist(win_flat, lib_feats)  # [N*W, K]
                matched_idx_flat = torch.argmin(dist, dim=1)  # [N*W]
                token_idx_windows = matched_idx_flat.view(N_ships_eff, num_windows)
            
            # 需要将token_idx扩展到每个时间步
            # 方法：每个token重复shift次
            token_idx = torch.zeros(num_ships, num_steps, dtype=torch.int64)
            
            # 填充token索引（每shift步重复同一个token）——与窗口提取时的 S_lib 上界保持一致
            for i, t in enumerate(window_starts):
                # 将这个token复制到对应的时间步范围
                for j in range(self.shift):
                    if t + j < num_steps:
                        token_idx[:, t + j] = token_idx_windows[:, i]
            
            # 填充最后一个token到剩余步
            if num_windows > 0:
                last_token = token_idx_windows[:, -1]
                for t in range(start_t + num_windows * self.shift, num_steps):
                    token_idx[:, t] = last_token
        else:
            # 如果没有足够的步数，返回占位符
            token_idx = torch.zeros(num_ships, num_steps, dtype=torch.int64)
        
        return token_idx
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_historical_steps={self.num_historical_steps}, '
                f'num_future_steps={self.num_future_steps}, '
                f'mode={self.mode})')


if __name__ == "__main__":
    # 测试代码
    import os
    import sys
    sys.path.insert(0, '/home/mahexing/SMART-main')
    
    from smart.datasets.maritime_dataset import MaritimeDataset
    
    print("测试 MaritimeTargetBuilder...")
    
    # 创建transform（使用30秒时间步配置）
    transform = MaritimeTargetBuilder(
        num_historical_steps=5,
        num_future_steps=16,
        mode='train'
    )
    
    # 加载一个样本
    dataset = MaritimeDataset(
        root=None,
        split='train',
        raw_dir=['/home/mahexing/SMART-main/data/maritime_windows_v1/train'],
        transform=transform
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试transform
    sample = dataset[0]
    print(f"\n✅ Transform应用成功!")
    print(f"Agent特征形状: {sample['agent'].x.shape}")
    print(f"节点数: {sample['agent'].num_nodes}")
    print(f"AV索引: {sample['agent'].av_index}")

