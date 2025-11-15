#!/usr/bin/env python3
"""
全部Token彩色可视化 - 包含所有2048个向量
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_all_colored_vectors(traj, output_path):
    """
    使用鲜明颜色可视化所有向量
    """
    print(f"\n生成全部向量彩色图（{len(traj)}个向量）...")
    
    # 使用所有向量
    start_points = traj[:, 0, :2]
    end_points = traj[:, 1, :2]
    vectors = end_points - start_points
    
    print(f"向量范围: X[{vectors[:, 0].min():.1f}, {vectors[:, 0].max():.1f}], Y[{vectors[:, 1].min():.1f}, {vectors[:, 1].max():.1f}]")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('lightgray')  # 浅灰色背景
    
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
    print("绘制箭头...")
    for i in range(len(vectors)):
        ax.arrow(origin[i, 0], origin[i, 1], 
                 vectors[i, 0], vectors[i, 1],
                 head_width=6, head_length=10,
                 fc=colors[i], ec='black', linewidth=0.8,
                 alpha=0.6,
                 length_includes_head=True)
        
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(vectors)}")
    
    print("  进度: 完成！")
    
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
    ax.set_title(f'Maritime Token Vectors - All {len(vectors)} Tokens\nColored by Direction (from origin)', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('X Displacement (meters)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Displacement (meters)', fontsize=18, fontweight='bold')
    
    # 统计信息
    vector_lengths = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # 统计各方向的数量
    east_count = sum(1 for c in colors if c == 'red')
    west_count = sum(1 for c in colors if c == 'blue')
    north_count = sum(1 for c in colors if c == 'green')
    south_count = sum(1 for c in colors if c == 'orange')
    
    info_text = f'Total tokens: {len(vectors)}\n'
    info_text += f'Avg length: {vector_lengths.mean():.1f}m\n'
    info_text += f'Range: [{vector_lengths.min():.1f}, {vector_lengths.max():.1f}]m\n\n'
    info_text += f'Direction distribution:\n'
    info_text += f'East (Red):    {east_count:4d} ({100*east_count/len(vectors):.1f}%)\n'
    info_text += f'North (Green): {north_count:4d} ({100*north_count/len(vectors):.1f}%)\n'
    info_text += f'West (Blue):   {west_count:4d} ({100*west_count/len(vectors):.1f}%)\n'
    info_text += f'South (Orange):{south_count:4d} ({100*south_count/len(vectors):.1f}%)'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
            fontweight='bold', family='monospace')
    
    print("保存图片...")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("🎨 Maritime Token全部2048向量彩色可视化")
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
    
    # 生成全部向量彩色版图
    visualize_all_colored_vectors(traj, output_dir / "token_all_vectors_colored.png")
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"\n生成的图片：assets/token_all_vectors_colored.png")
    print()


if __name__ == "__main__":
    main()

