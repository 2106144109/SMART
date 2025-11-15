#!/usr/bin/env python3
"""
为30秒间隔数据生成高多样性Maritime轨迹词汇表
优化配置：增加动作多样性
"""

import os
import sys
sys.path.insert(0, '/home/mahexing/SMART-main')

from maritime_traj_clustering import create_maritime_vocabulary

if __name__ == '__main__':
    
    print("\n" + "=" * 80)
    print("🎨 高多样性Maritime轨迹聚类配置")
    print("=" * 80)
    print("\n选择配置方案:")
    print("  1. 中等多样性 (3072 tokens, +50%多样性)")
    print("  2. 高多样性 (4096 tokens, +80%多样性, 推荐⭐)")
    print("  3. 极限多样性 (8192 tokens, +100%多样性)")
    print("  4. 默认配置 (2048 tokens, 基础)")
    print()
    
    choice = input("请选择 (1-4) [默认: 2]: ").strip() or "2"
    
    # 预设配置方案
    configs = {
        "1": {
            'name': '中等多样性',
            'num_clusters': 3072,
            'tolerance': 0.12,
            'shift': 2,
            'max_samples': 50000,
        },
        "2": {
            'name': '高多样性',
            'num_clusters': 4096,
            'tolerance': 0.10,
            'shift': 3,
            'max_samples': 100000,
        },
        "3": {
            'name': '极限多样性',
            'num_clusters': 8192,
            'tolerance': 0.08,
            'shift': 4,
            'max_samples': -1,  # 使用全部数据
        },
        "4": {
            'name': '默认配置',
            'num_clusters': 2048,
            'tolerance': 0.15,
            'shift': 2,
            'max_samples': 50000,
        }
    }
    
    if choice not in configs:
        print(f"无效选择，使用默认方案2")
        choice = "2"
    
    selected = configs[choice]
    
    # 完整配置
    config = {
        'data_dirs': [
            '/home/mahexing/SMART-main/data/maritime_windows_30s/train',
        ],
        'num_clusters': selected['num_clusters'],
        'shift': selected['shift'],
        'max_samples': selected['max_samples'],
        'ship_width': 10.0,
        'ship_length': 50.0,
        'tolerance': selected['tolerance'],
        'output_path': f'data/maritime_motion_vocab_30s_diverse_{choice}.pt'
    }
    
    # 显示配置
    print("\n" + "=" * 80)
    print(f"📋 配置方案: {selected['name']}")
    print("=" * 80)
    print(f"\n⚙️ 参数设置:")
    print(f"  数据源: {config['data_dirs'][0]}")
    print(f"  词汇表大小: {config['num_clusters']} tokens")
    print(f"  容差: {config['tolerance']}")
    print(f"  时间步数: {config['shift']} (每token覆盖 {(config['shift']+1)*30}秒)")
    print(f"  最大样本数: {config['max_samples'] if config['max_samples'] > 0 else '全部'}")
    print(f"  船舶尺寸: {config['ship_width']}m × {config['ship_length']}m")
    print(f"  输出路径: {config['output_path']}")
    
    # 预估时间
    time_estimates = {
        "1": "15-30分钟",
        "2": "30-60分钟",
        "3": "1-2小时",
        "4": "10-20分钟"
    }
    print(f"\n⏱️ 预计用时: {time_estimates[choice]}")
    
    # 检查数据目录
    if not os.path.exists(config['data_dirs'][0]):
        print(f"\n❌ 错误：数据目录不存在！")
        print(f"   请先运行: bash regenerate_30s_data.sh")
        sys.exit(1)
    
    # 确认
    print("\n" + "=" * 80)
    confirm = input("确认开始生成? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        sys.exit(0)
    
    # 创建词汇表
    print("\n🚀 开始生成词汇表...")
    print("   请耐心等待，进度会定期显示...")
    print()
    
    vocab = create_maritime_vocabulary(**config)
    
    print("\n" + "=" * 80)
    print("✅ 词汇表生成完成！")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"   {config['output_path']}")
    print(f"\n📊 多样性分析:")
    
    # 简单的多样性分析
    import torch
    import numpy as np
    
    traj = vocab['traj']['ship'].numpy()  # [N, time_steps, 3]
    
    # 计算速度分布
    dx = np.diff(traj[:, :, 0], axis=1)
    dy = np.diff(traj[:, :, 1], axis=1)
    speeds = np.sqrt(dx**2 + dy**2).mean(axis=1)
    
    # 计算转向分布
    dtheta = np.diff(traj[:, :, 2], axis=1)
    turns = np.abs(dtheta).mean(axis=1)
    
    print(f"   Token数量: {len(traj)}")
    print(f"   速度范围: [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
    print(f"   速度标准差: {speeds.std():.2f}")
    print(f"   转向范围: [{turns.min():.3f}, {turns.max():.3f}] rad")
    print(f"   转向标准差: {turns.std():.3f}")
    
    # 分类统计
    slow = (speeds < 2).sum()
    medium = ((speeds >= 2) & (speeds < 5)).sum()
    fast = (speeds >= 5).sum()
    
    print(f"\n   速度分布:")
    print(f"     慢速(<2m/s): {slow} ({slow/len(traj)*100:.1f}%)")
    print(f"     中速(2-5m/s): {medium} ({medium/len(traj)*100:.1f}%)")
    print(f"     快速(>5m/s): {fast} ({fast/len(traj)*100:.1f}%)")
    
    straight = (turns < 0.1).sum()
    turn = (turns >= 0.1).sum()
    
    print(f"\n   转向分布:")
    print(f"     直行(<0.1rad): {straight} ({straight/len(traj)*100:.1f}%)")
    print(f"     转弯(≥0.1rad): {turn} ({turn/len(traj)*100:.1f}%)")
    
    print(f"\n🎯 下一步:")
    print(f"   1. 转换为token格式:")
    print(f"      python convert_vocab_to_token.py \\")
    print(f"        --vocab {config['output_path']} \\")
    print(f"        --output smart/tokens/maritime_tokens_30s_diverse_{choice}.pkl")
    print(f"\n   2. 使用新词典训练:")
    print(f"      - 复制token文件为 maritime_tokens.pkl")
    print(f"      - 或修改模型代码指向新token文件")
    print()



