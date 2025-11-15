#!/usr/bin/env python3
"""
可视化Maritime Token词典
生成三个分析图：
1. 所有轨迹向量图（从原点出发）
2. 位移大小分布图
3. 方向分布图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体 - 使用系统中可用的字体
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

def load_tokens(token_path):
    """加载Token文件"""
    print(f"📂 加载Token文件: {token_path}")
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取ship的轨迹数据
    traj = data['traj']['ship']  # (N, 2, 3) - N个token，2个时间点，(x, y, theta)
    
    print(f"✅ 加载完成")
    print(f"   Token数量: {traj.shape[0]}")
    print(f"   时间步数: {traj.shape[1]}")
    print(f"   特征维度: {traj.shape[2]} (x, y, theta)")
    
    return traj, data


def visualize_all_vectors(traj, output_path):
    """
    图1: 所有向量从原点出发的图
    显示所有token的运动向量
    """
    print(f"\n🎨 生成图1: 所有轨迹向量图...")
    
    # 计算起点到终点的向量
    start_points = traj[:, 0, :2]  # (N, 2) - 起点(x, y)
    end_points = traj[:, 1, :2]    # (N, 2) - 终点(x, y)
    vectors = end_points - start_points  # (N, 2) - 位移向量
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 从原点绘制所有向量
    origin = np.zeros((len(vectors), 2))
    
    # 使用颜色映射表示向量长度
    vector_lengths = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # 绘制向量
    quiver = ax.quiver(origin[:, 0], origin[:, 1], 
                       vectors[:, 0], vectors[:, 1],
                       vector_lengths,
                       cmap='viridis',
                       alpha=0.6,
                       scale_units='xy',
                       scale=1,
                       width=0.003)
    
    # 添加颜色条
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('位移大小 (m)', fontsize=12)
    
    # 设置坐标轴
    max_range = max(np.abs(vectors).max(), 150)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # 添加方向指示
    ax.text(max_range * 0.9, 0, '东', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(-max_range * 0.9, 0, '西', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(0, max_range * 0.9, '北', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(0, -max_range * 0.9, '南', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # 标题和标签
    ax.set_title(f'Maritime Token轨迹向量图 (共{len(vectors)}个Token)\n所有向量从原点出发', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X位移 (m)', fontsize=14)
    ax.set_ylabel('Y位移 (m)', fontsize=14)
    
    # 添加统计信息
    info_text = f'平均位移: {vector_lengths.mean():.1f}m\n'
    info_text += f'最大位移: {vector_lengths.max():.1f}m\n'
    info_text += f'最小位移: {vector_lengths.min():.1f}m'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {output_path}")
    plt.close()


def visualize_displacement_distribution(traj, output_path):
    """
    图2: 位移大小分布图
    """
    print(f"\n🎨 生成图2: 位移大小分布图...")
    
    # 计算位移
    start_points = traj[:, 0, :2]
    end_points = traj[:, 1, :2]
    vectors = end_points - start_points
    displacements = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：直方图
    n, bins, patches = ax1.hist(displacements, bins=50, 
                                  color='steelblue', 
                                  edgecolor='black', 
                                  alpha=0.7)
    
    # 根据高度着色
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax1.axvline(displacements.mean(), color='red', linestyle='--', linewidth=2, label=f'平均值: {displacements.mean():.1f}m')
    ax1.axvline(np.median(displacements), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(displacements):.1f}m')
    ax1.set_xlabel('位移大小 (m)', fontsize=14)
    ax1.set_ylabel('Token数量', fontsize=14)
    ax1.set_title('位移大小分布直方图', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 右图：累积分布
    sorted_disp = np.sort(displacements)
    cumulative = np.arange(1, len(sorted_disp) + 1) / len(sorted_disp) * 100
    
    ax2.plot(sorted_disp, cumulative, linewidth=2, color='steelblue')
    ax2.axhline(50, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # 标注关键百分位
    p50 = np.percentile(displacements, 50)
    p95 = np.percentile(displacements, 95)
    ax2.axvline(p50, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(p95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(p50, 55, f'P50: {p50:.1f}m', fontsize=10, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.text(p95, 97, f'P95: {p95:.1f}m', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlabel('位移大小 (m)', fontsize=14)
    ax2.set_ylabel('累积百分比 (%)', fontsize=14)
    ax2.set_title('位移累积分布曲线', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'统计信息:\n'
    stats_text += f'样本数: {len(displacements)}\n'
    stats_text += f'平均值: {displacements.mean():.2f} m\n'
    stats_text += f'标准差: {displacements.std():.2f} m\n'
    stats_text += f'最小值: {displacements.min():.2f} m\n'
    stats_text += f'最大值: {displacements.max():.2f} m\n'
    stats_text += f'P25: {np.percentile(displacements, 25):.2f} m\n'
    stats_text += f'P50: {np.percentile(displacements, 50):.2f} m\n'
    stats_text += f'P75: {np.percentile(displacements, 75):.2f} m\n'
    stats_text += f'P95: {np.percentile(displacements, 95):.2f} m'
    
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"✅ 保存: {output_path}")
    plt.close()


def visualize_direction_distribution(traj, output_path):
    """
    图3: 方向分布图
    包含极坐标图和象限统计
    """
    print(f"\n🎨 生成图3: 方向分布图...")
    
    # 计算方向
    start_points = traj[:, 0, :2]
    end_points = traj[:, 1, :2]
    vectors = end_points - start_points
    directions = np.arctan2(vectors[:, 1], vectors[:, 0])  # 弧度，范围[-π, π]
    
    # 创建图形
    fig = plt.figure(figsize=(18, 6))
    
    # 左图：极坐标直方图
    ax1 = plt.subplot(131, projection='polar')
    
    # 将方向转换到[0, 2π]
    directions_positive = directions.copy()
    directions_positive[directions_positive < 0] += 2 * np.pi
    
    # 绘制极坐标直方图
    n_bins = 36  # 每10度一个bin
    theta_bins = np.linspace(0, 2*np.pi, n_bins + 1)
    radii, _ = np.histogram(directions_positive, bins=theta_bins)
    theta = (theta_bins[:-1] + theta_bins[1:]) / 2
    width = 2 * np.pi / n_bins
    
    bars = ax1.bar(theta, radii, width=width, bottom=0, alpha=0.7)
    
    # 根据高度着色
    cm = plt.cm.viridis
    for r, bar in zip(radii, bars):
        bar.set_facecolor(cm(r / radii.max()))
        bar.set_edgecolor('black')
    
    ax1.set_theta_zero_location('E')  # 0度在东方（右侧）
    ax1.set_theta_direction(1)  # 逆时针
    ax1.set_title('方向分布极坐标图\n(0°=东, 90°=北)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True)
    
    # 中图：方向直方图（角度）
    ax2 = plt.subplot(132)
    
    # 转换为角度
    directions_deg = np.degrees(directions)
    
    n, bins, patches = ax2.hist(directions_deg, bins=36, 
                                  range=(-180, 180),
                                  color='steelblue', 
                                  edgecolor='black', 
                                  alpha=0.7)
    
    # 着色
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = n / n.max()
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # 标注关键方向
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='东(0°)')
    ax2.axvline(90, color='green', linestyle='--', linewidth=1, alpha=0.7, label='北(90°)')
    ax2.axvline(-90, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='南(-90°)')
    ax2.axvline(180, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(-180, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='西(±180°)')
    
    ax2.set_xlabel('方向 (度)', fontsize=14)
    ax2.set_ylabel('Token数量', fontsize=14)
    ax2.set_title('方向分布直方图', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 右图：象限统计
    ax3 = plt.subplot(133)
    
    # 计算各象限数量
    Q1 = np.sum((directions >= 0) & (directions < np.pi/2))      # 东北
    Q2 = np.sum((directions >= np.pi/2) & (directions <= np.pi)) # 西北
    Q3 = np.sum((directions >= -np.pi) & (directions < -np.pi/2)) # 西南
    Q4 = np.sum((directions >= -np.pi/2) & (directions < 0))     # 东南
    
    quadrants = ['Q1\n东北\n(0°-90°)', 'Q2\n西北\n(90°-180°)', 
                 'Q3\n西南\n(-180°--90°)', 'Q4\n东南\n(-90°-0°)']
    counts = [Q1, Q2, Q3, Q4]
    percentages = [c / len(directions) * 100 for c in counts]
    
    # 绘制柱状图
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax3.bar(range(4), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加百分比标注
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加期望值参考线
    expected = len(directions) / 4
    ax3.axhline(expected, color='red', linestyle='--', linewidth=2, 
                label=f'期望值: {expected:.0f} (25%)')
    
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(quadrants, fontsize=11)
    ax3.set_ylabel('Token数量', fontsize=14)
    ax3.set_title('象限分布统计', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 计算卡方检验
    chi_square = sum((c - expected)**2 / expected for c in counts)
    
    # 添加统计信息
    stats_text = f'卡方值: {chi_square:.2f}\n'
    stats_text += f'(< 50为良好均匀性)\n'
    if chi_square < 50:
        stats_text += '✅ 分布均匀'
    else:
        stats_text += '⚠️ 分布不均'
    
    ax3.text(0.5, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("🎨 Maritime Token词典可视化")
    print("=" * 80)
    
    # 文件路径
    token_path = Path("smart/tokens/maritime_tokens_no_norm.pkl")
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    traj, data = load_tokens(token_path)
    
    # 生成三个图
    visualize_all_vectors(traj, output_dir / "token_all_vectors.png")
    visualize_displacement_distribution(traj, output_dir / "token_displacement_dist.png")
    visualize_direction_distribution(traj, output_dir / "token_direction_dist.png")
    
    print("\n" + "=" * 80)
    print("✅ 可视化完成！")
    print("=" * 80)
    print(f"\n生成的图片：")
    print(f"  1. {output_dir / 'token_all_vectors.png'}")
    print(f"  2. {output_dir / 'token_displacement_dist.png'}")
    print(f"  3. {output_dir / 'token_direction_dist.png'}")
    print()


if __name__ == "__main__":
    main()

