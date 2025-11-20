#!/usr/bin/env python3
"""
Maritime Dataset for SMART
专门用于加载海上场景的 HeteroData 数据集
"""

import os
from collections import OrderedDict
import torch
from typing import Any, Callable, List, Mapping, Optional
from torch_geometric.data import Dataset, HeteroData
from smart.utils.log import Logging


def _infer_num_nodes(store: Mapping[str, Any]) -> Optional[int]:
    """Best-effort inference for `num_nodes` from common tensor fields."""

    if "num_nodes" in store:
        try:
            return int(store["num_nodes"])
        except Exception:
            return store["num_nodes"]

    for key in ("position", "x", "token_idx", "traj_pos"):
        val = store.get(key)
        if isinstance(val, torch.Tensor) and val.dim() >= 1:
            return val.shape[0]
    return None


def _dict_to_heterodata(data: Mapping[str, Any]) -> HeteroData:
    """Convert a plain dict sample into HeteroData for downstream modules."""

    hd = HeteroData()
    for key, value in data.items():
        # Edge stores: (src, rel, dst)
        if isinstance(key, tuple) and len(key) == 3:
            if isinstance(value, Mapping):
                hd[key].update(value)
            else:
                hd[key] = value
            continue

        # Global attributes (非节点特征)
        if key in {"map_save", "city"}:
            hd[key] = value
            continue

        # Node stores
        if isinstance(value, Mapping):
            hd[key].update(value)
            num_nodes = _infer_num_nodes(value)
            if num_nodes is not None and "num_nodes" not in hd[key]:
                hd[key]["num_nodes"] = num_nodes
            continue

        # Fallback to simple assignment
        hd[key] = value

    # Maritime dict samples may lack `x`; try to synthesize it from basic fields.
    if "agent" in hd.node_types:
        agent = hd["agent"]
        if not hasattr(agent, "x"):
            position = agent.get("position") if isinstance(agent, Mapping) else None
            velocity = agent.get("velocity") if isinstance(agent, Mapping) else None
            acceleration = agent.get("acceleration") if isinstance(agent, Mapping) else None
            heading = agent.get("heading") if isinstance(agent, Mapping) else None
            omega = agent.get("omega") if isinstance(agent, Mapping) else None

            if isinstance(position, torch.Tensor) and position.dim() >= 2:
                num_agents, num_steps = position.shape[:2]
                dtype = position.dtype
                device = position.device
                x = torch.zeros((num_agents, num_steps, 8), dtype=dtype, device=device)

                # xy positions
                x[:, :, 0:2] = position[:, :, 0:2]

                # velocities (vx, vy)
                if isinstance(velocity, torch.Tensor) and velocity.shape[:2] == (num_agents, num_steps):
                    x[:, :, 2:4] = velocity[:, :, 0:2]

                # accelerations (ax, ay)
                if isinstance(acceleration, torch.Tensor) and acceleration.shape[:2] == (num_agents, num_steps):
                    x[:, :, 4:6] = acceleration[:, :, 0:2]

                # heading (theta)
                if isinstance(heading, torch.Tensor):
                    # broadcast to match [N, T]
                    x[:, :, 6] = heading.reshape(num_agents, num_steps)

                # angular velocity (omega)
                if isinstance(omega, torch.Tensor):
                    x[:, :, 7] = omega.reshape(num_agents, num_steps)

                agent["x"] = x

            # Ensure num_nodes is set if we built x
            if hasattr(agent, "x") and not hasattr(agent, "num_nodes"):
                agent["num_nodes"] = agent.x.shape[0]

    return hd


class MaritimeDataset(Dataset):
    """
    Maritime Scene Dataset
    直接加载预处理好的 .pt 文件（HeteroData格式）
    """
    
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: List[str] = None,
                 processed_dir: List[str] = None,
                 transform: Optional[Callable] = None,
                 **kwargs) -> None:
        """
        Args:
            root: 数据根目录（不使用，为了兼容）
            split: 'train', 'val', 或 'test'
            raw_dir: 数据目录列表，包含 .pt 文件
            processed_dir: 已处理数据目录（不使用，数据已是 .pt 格式）
            transform: 可选的数据变换（通常不需要，数据已预处理）
        """
        self.logger = Logging().log(level='DEBUG')
        self.split = split
        self.training = split == 'train'
        self.transform = transform
        
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        
        # 收集所有 .pt 文件路径
        self._file_paths = []
        if raw_dir is not None:
            for dir_path in raw_dir:
                dir_path = os.path.expanduser(os.path.normpath(dir_path))
                if not os.path.exists(dir_path):
                    self.logger.warning(f"目录不存在: {dir_path}")
                    continue
                
                # 获取所有 .pt 文件
                pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
                pt_files.sort()  # 确保顺序一致
                
                self._file_paths.extend([os.path.join(dir_path, f) for f in pt_files])
        
        # 构建样本索引：每个文件可能包含多个样本（列表格式）
        # 格式：[(file_idx, sample_idx_in_list), ...]
        self._sample_indices = []
        # LRU 文件缓存，减少频繁的 torch.load I/O
        self._cache_capacity = int(kwargs.get('cache_capacity', 16))
        self._file_cache: OrderedDict[int, object] = OrderedDict()
        
        self.logger.info(f"扫描 {split} 数据集文件...")
        
        for file_idx, file_path in enumerate(self._file_paths):
            try:
                # 快速加载检查是否为列表格式
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                
                if isinstance(data, list):
                    # 列表格式：每个元素是一个样本
                    num_samples_in_file = len(data)
                    for sample_idx in range(num_samples_in_file):
                        self._sample_indices.append((file_idx, sample_idx))
                else:
                    # 单个样本
                    self._sample_indices.append((file_idx, 0))
                
                # 释放内存（不缓存扫描时加载的数据）
                del data
                    
            except Exception as e:
                self.logger.warning(f"跳过文件 {file_path}: {e}")
                continue
        
        self._num_samples = len(self._sample_indices)
        self.logger.info(f"加载 {split} 数据集: {self._num_samples} 个样本 (来自 {len(self._file_paths)} 个文件)")
        
        # 不调用父类的 __init__，因为我们不需要它的处理逻辑
        # 直接初始化必要的属性
        self.root = root
        self._indices = None
        
    def len(self) -> int:
        """返回数据集大小"""
        return self._num_samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self._num_samples
    
    def get(self, idx: int):
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            HeteroData对象
        """
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self._num_samples})")
        
        # 获取样本的文件索引和列表内索引
        file_idx, sample_idx = self._sample_indices[idx]
        file_path = self._file_paths[file_idx]
        
        try:
            # 优先从缓存读取
            if file_idx in self._file_cache:
                data = self._file_cache.pop(file_idx)
                self._file_cache[file_idx] = data  # 触发最近使用
            else:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
                # 放入缓存并维持容量
                self._file_cache[file_idx] = data
                if self._cache_capacity > 0 and len(self._file_cache) > self._cache_capacity:
                    self._file_cache.popitem(last=False)
            
            # 如果是列表格式，提取指定索引的样本
            if isinstance(data, list):
                data = data[sample_idx]
            # 如果不是列表，sample_idx应该是0
            elif sample_idx != 0:
                raise ValueError(f"文件 {file_path} 不是列表格式，但索引为 {sample_idx}")

            # 兼容 dict 保存的样本：转换为 HeteroData
            if isinstance(data, dict):
                data = _dict_to_heterodata(data)

            # DEBUG: 在transform之前检查数据
            if not hasattr(data, 'node_types'):
                raise ValueError(f"加载的数据不是HeteroData: type={type(data)}, file={file_path}, idx={sample_idx}")
            
            if 'agent' not in data.node_types:
                actual_nodes = list(data.node_types)
                raise ValueError(f"加载的数据缺少agent节点: nodes={actual_nodes}, file={file_path}, idx={sample_idx}")
            
            # 应用可选的transform（如果有）
            if self.transform is not None:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载样本失败 (文件: {file_path}, 索引: {sample_idx}): {e}")
            raise
    
    def __getitem__(self, idx: int):
        """支持索引访问"""
        return self.get(idx)
    
    @property
    def raw_file_names(self):
        """原始文件名（为了兼容 PyG Dataset）"""
        return [os.path.basename(p) for p in self._file_paths]
    
    @property
    def processed_file_names(self):
        """处理后的文件名（为了兼容 PyG Dataset）"""
        return []
    
    def download(self):
        """下载数据（不需要）"""
        pass
    
    def process(self):
        """处理数据（不需要，数据已预处理）"""
        pass


if __name__ == "__main__":
    # 测试代码
    dataset = MaritimeDataset(
        root=None,
        split='train',
        raw_dir=['/home/mahexing/SMART-main/data/maritime_windows_v1/train']
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"加载第一个样本...")
    
    sample = dataset[0]
    print(f"\n样本结构:")
    print(f"  节点类型: {sample.node_types}")
    print(f"  边类型: {sample.edge_types}")
    
    if 'agent' in sample:
        print(f"\nAgent信息:")
        print(f"  特征形状: {sample['agent'].x.shape}")
        print(f"  节点数: {sample['agent'].num_nodes}")
        if hasattr(sample['agent'], 'valid_mask'):
            print(f"  有效掩码: {sample['agent'].valid_mask.shape}")
        if hasattr(sample['agent'], 'mmsi'):
            print(f"  MMSI数量: {len(sample['agent'].mmsi)}")
    
    print("\n✅ Maritime数据集测试成功!")

