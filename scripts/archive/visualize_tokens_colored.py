#!/usr/bin/env python3
"""
彩色版Token可视化 - 使用明显的颜色
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_colored_vectors(traj, output_path, num_samples=200):
    """
    使用鲜明颜色可视化向量
    """
    print(f"\n生成彩色版向量图（{num_samples}个样本）...")
    
    # 随机采样
    indices = np.random.choice(len(traj), min(num_samples, len(traj)), replace=False)
    traj_sample = traj[indices]
    
    # 计算向量
    start_points = traj_sample[:, 0, :2]
    end_points = traj_sample[:, 1, :2]
    vectors = end_points - start_points
    
    print(f"向量范围: X[{vectors[:, 0].min():.1f}, {vectors[:, 0].max():.1f}], Y[{vectors[:, 1].min():.1f}, {vectors[:, 1].max():.1f}]")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('lightgray')  # 浅灰色背景，更容易看到箭头
    
    # 从原点绘制向量
    origin = np.zeros((len(vectors), 2))
    
    # 计算每个向量的方向角，用于着色
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 根据方向角分配颜色
    colors = []
    for angle in angles:
        angle_deg = np.degrees(angle)
        if -45 <= angle_deg < 45:
            colors.append('red')  # 东
        elif 45 <= angle_deg < 135:
            colors.append('green')  # 北
        elif angle_deg >= 135 or angle_deg < -135:
            colors.append('blue')  # 西
        else:
            colors.append('orange')  # 南
    
    # 绘制每个箭头
    for i in range(len(vectors)):
        ax.arrow(origin[i, 0], origin[i, 1], 
                 vectors[i, 0], vectors[i, 1],
                 head_width=8, head_length=12,
                 fc=colors[i], ec='black', linewidth=1.5,
                 alpha=0.7,
                 length_includes_head=True)
    
    # 设置坐标轴
    max_range = 180
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.6, linewidth=1.5, color='white')
    ax.axhline(y=0, color='black', linewidth=3)
    ax.axvline(x=0, color='black', linewidth=3)
    
    # 添加方向标注
    ax.text(max_range * 0.85, 0, 'EAST\n(Red)', ha='center', va='center', 
            fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax.text(-max_range * 0.85, 0, 'WEST\n(Blue)', ha='center', va='center', 
            fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax.text(0, max_range * 0.85, 'NORTH\n(Green)', ha='center', va='center', 
            fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax.text(0, -max_range * 0.85, 'SOUTH\n(Orange)', ha='center', va='center', 
            fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # 标题
    ax.set_title(f'Maritime Token Vectors - Colored by Direction\n({len(vectors)} samples, all from origin)', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('X Displacement (meters)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Displacement (meters)', fontsize=18, fontweight='bold')
    
    # 统计信息
    vector_lengths = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    info_text = f'Total vectors: {len(vectors)}\n'
    info_text += f'Avg length: {vector_lengths.mean():.1f}m\n'
    info_text += f'Range: [{vector_lengths.min():.1f}, {vector_lengths.max():.1f}]m\n\n'
    info_text += f'Color code:\n'
    info_text += f'Red = East, Blue = West\n'
    info_text += f'Green = North, Orange = South'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("🎨 Maritime Token彩色版可视化")
    print("=" * 80)
    
    # 文件路径
    token_path = Path("smart/tokens/maritime_tokens_no_norm.pkl")
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    print(f"📂 加载Token文件: {token_path}")
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
    
    traj = data['traj']['ship']
    print(f"✅ 加载完成")
    print(f"   Token数量: {traj.shape[0]}")
    print(f"   时间步数: {traj.shape[1]}")
    
    # 生成彩色版图
    visualize_colored_vectors(traj, output_dir / "token_vectors_colored.png", num_samples=200)
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"\n生成的图片：")
    print(f"  1. test_red_arrows.png (10个红色箭头测试)")
    print(f"  2. assets/token_vectors_colored.png (200个彩色箭头)")
    print()


if __name__ == "__main__":
    main()

