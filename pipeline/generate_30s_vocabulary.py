#!/usr/bin/env python3
"""
为30秒间隔数据生成Maritime轨迹词汇表
"""

import os
import sys
sys.path.insert(0, '/home/mahexing/SMART-main')

from maritime_traj_clustering import create_maritime_vocabulary

if __name__ == '__main__':
    # ========== 配置参数 ==========
    # 词汇表大小（token数量）
    NUM_CLUSTERS = 512  # ← 修改这里！推荐值: 256/512/1024/2048
    
    # 容差设置（调整聚类精细度）
    # 推荐值：0.15(默认) / 0.12(+20%多样性) / 0.10(+40%多样性)
    TOLERANCE = 0.10  # ← 修改这里来调整多样性
    # ================================
    
    # 30秒间隔数据的配置参数
    config = {
        'data_dirs': [
            '/home/mahexing/SMART-main/data/maritime_windows_30s/train',
        ],
        'num_clusters': NUM_CLUSTERS,  # 词汇表大小
        'shift': 1,                    # 时间步数（2个点，30秒）
        'max_samples': 50000,          # 最大使用样本数（保持不变）
        'ship_width': 10.0,            # 船舶宽度（保持不变）
        'ship_length': 50.0,           # 船舶长度（保持不变）
        'tolerance': TOLERANCE,        # 聚类容差
        'output_path': f'data/maritime_motion_vocab_30s_{NUM_CLUSTERS}tokens_shift{1}_tol{TOLERANCE:.2f}.pt'
    }
    
    print("\n" + "=" * 80)
    print("🚢 Maritime轨迹聚类脚本 (30秒间隔)")
    print("=" * 80)
    print("\n⚙️ 配置参数:")
    print(f"  数据源: {config['data_dirs'][0]}")
    print(f"  ⭐ 词汇表大小: {NUM_CLUSTERS} tokens")
    print(f"  ⭐ 容差: {TOLERANCE}")
    print(f"  ⭐ 时间步数 shift: {config['shift']} (2个点, 30秒轨迹)")
    print(f"  最大样本数: {config['max_samples']} (保持不变)")
    print(f"  输出路径: {config['output_path']}")
    
    # 词汇表大小说明
    vocab_hints = {
        256: "小词汇表 - 适合简单场景/快速训练",
        512: "中等词汇表 - 平衡效率和表达力",
        1024: "较大词汇表 - 适合中等复杂度",
        2048: "大词汇表 - 标准配置，高表达力",
        4096: "超大词汇表 - 极高表达力"
    }
    vocab_hint = vocab_hints.get(NUM_CLUSTERS, f"{NUM_CLUSTERS} tokens (自定义)")
    print(f"\n  配置说明: {vocab_hint}")
    
    # 容差说明
    diversity_hint = {
        0.15: "标准多样性",
        0.12: "+20%多样性",
        0.10: "+40%多样性",
        0.08: "+60%多样性",
    }
    hint = diversity_hint.get(TOLERANCE, "自定义设置")
    print(f"  容差效果: {hint}")
    
    # 检查数据目录
    if not os.path.exists(config['data_dirs'][0]):
        print(f"\n❌ 错误：数据目录不存在！")
        print(f"   请先运行: bash regenerate_30s_data.sh")
        sys.exit(1)
    
    # 创建词汇表
    print("\n🚀 开始生成词汇表...")
    print("   这可能需要10-30分钟，请耐心等待...")
    print()
    
    vocab = create_maritime_vocabulary(**config)
    
    print("\n" + "=" * 80)
    print("✅ 词汇表生成完成！")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"   {config['output_path']}")
    print(f"\n🎯 下一步:")
    print(f"   运行转换脚本:")
    print(f"   python convert_vocab_to_token.py \\")
    print(f"     --vocab {config['output_path']} \\")
    print(f"     --output smart/tokens/maritime_tokens_30s.pkl")
    print()

