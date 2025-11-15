#!/usr/bin/env python3
"""
将Maritime词汇表转换为SMART模型使用的token格式
"""

import torch
import pickle
import numpy as np
from pathlib import Path


def convert_maritime_vocab_to_token(vocab_path, output_path):
    """
    将maritime_motion_vocab.pt转换为SMART模型使用的pkl格式
    
    Args:
        vocab_path: 输入的词汇表路径 (maritime_motion_vocab.pt)
        output_path: 输出的token文件路径 (*.pkl)
    """
    print("=" * 80)
    print("🔄 转换Maritime词汇表为Token格式")
    print("=" * 80)
    
    # 加载maritime词汇表
    print(f"\n📂 加载词汇表: {vocab_path}")
    vocab = torch.load(vocab_path, map_location='cpu', weights_only=False)
    
    # 提取ship的token和traj
    if 'token' not in vocab or 'ship' not in vocab['token']:
        raise ValueError("词汇表格式不正确，缺少 token['ship']")
    
    if 'traj' not in vocab or 'ship' not in vocab['traj']:
        raise ValueError("词汇表格式不正确，缺少 traj['ship']")
    
    if 'token_all' not in vocab or 'ship' not in vocab['token_all']:
        raise ValueError("词汇表格式不正确，缺少 token_all['ship']")
    
    ship_token = vocab['token']['ship']          # [N, 4, 2]
    ship_traj = vocab['traj']['ship']            # [N, time_steps, 3]
    ship_token_all = vocab['token_all']['ship']  # [N, time_steps, 4, 2]
    
    print(f"   Token形状: {ship_token.shape}")
    print(f"   Traj形状: {ship_traj.shape}")
    print(f"   Token_all形状: {ship_token_all.shape}")
    
    # 转换为numpy（如果是tensor）
    if isinstance(ship_token, torch.Tensor):
        ship_token = ship_token.numpy()
    if isinstance(ship_traj, torch.Tensor):
        ship_traj = ship_traj.numpy()
    if isinstance(ship_token_all, torch.Tensor):
        ship_token_all = ship_token_all.numpy()
    
    # SMART模型期望的格式（从原始代码推断）:
    # token_data = {
    #     'token': dict with keys like 'veh', 'cyc', 'ped'
    #     'traj': dict with corresponding trajectories
    #     'token_all': dict with full time sequence polygons
    # }
    
    # 对于maritime场景，我们统一使用'veh'作为键（视船舶为车辆）
    # 或者创建单独的'ship'类别
    # Maritime场景只有ship，但模型需要veh/ped/cyc三种类型
    # 将ship数据复用到所有三种类型
    token_data = {
        'token': {
            'ship': ship_token,
            'veh': ship_token,   # 车辆（复用ship）
            'ped': ship_token,   # 行人（复用ship）
            'cyc': ship_token,   # 自行车（复用ship）
        },
        'traj': {
            'ship': ship_traj,
            'veh': ship_traj,
            'ped': ship_traj,
            'cyc': ship_traj,
        },
        'token_all': {
            'ship': ship_token_all,
            'veh': ship_token_all,
            'ped': ship_token_all,
            'cyc': ship_token_all,
        },
        'metadata': vocab.get('metadata', {})
    }
    
    # 保存为pkl格式
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(token_data, f)
    
    print(f"\n💾 Token文件已保存: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 验证
    print(f"\n✅ 验证保存的文件...")
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"   Token键: {list(loaded_data['token'].keys())}")
    print(f"   Traj键: {list(loaded_data['traj'].keys())}")
    print(f"   Token_all键: {list(loaded_data['token_all'].keys())}")
    
    for key in loaded_data['token'].keys():
        print(f"   {key} token形状: {loaded_data['token'][key].shape}")
        print(f"   {key} traj形状: {loaded_data['traj'][key].shape}")
        print(f"   {key} token_all形状: {loaded_data['token_all'][key].shape}")
    
    print(f"\n" + "=" * 80)
    print("✅ 转换完成！")
    print("=" * 80)
    
    return token_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='转换Maritime词汇表为Token格式')
    parser.add_argument('--vocab', type=str, 
                       default='data/maritime_motion_vocab.pt',
                       help='输入词汇表路径')
    parser.add_argument('--output', type=str,
                       default='smart/tokens/maritime_tokens.pkl',
                       help='输出token文件路径')
    
    args = parser.parse_args()
    
    convert_maritime_vocab_to_token(args.vocab, args.output)
    
    print("\n📝 下一步:")
    print(f"   1. Token文件已保存到: {args.output}")
    print(f"   2. 在训练配置中确保使用正确的token路径")
    print(f"   3. 开始训练: python train.py --config configs/train/train_maritime.yaml")

