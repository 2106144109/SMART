#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化运动词典：将全部 token 以位移向量绘制在同一张图上。
- 向量定义（默认）：每个 token 的代表轨迹从首帧到末帧的 (dx, dy) 位移（局部坐标，常见 Δy≈0）
- 可选：按末帧航向 θ 将位移模长 |Δr| 投影到全局方向（dx=|Δr|cosθ, dy=|Δr|sinθ），展示方向分布
- 展示方式：所有箭头从同一原点出发，长度和方向随 (dx, dy)

运行示例：
  python visualize_token_vocabulary.py \
    --vocab smart/tokens/maritime_tokens_no_norm.pkl \
    --output assets/token_vocab.png
"""

import os
import argparse
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')  # 无显示环境时写文件
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except Exception as e:
    raise RuntimeError("需要 matplotlib: pip install matplotlib") from e


def load_vocab(path: str):
    v = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(v, dict):
        raise ValueError('不支持的词典文件格式')
    traj_dict = v.get('traj')
    if not isinstance(traj_dict, dict) or len(traj_dict) == 0:
        raise ValueError("词典缺少 'traj' 键或为空")
    # 选择库键：优先 ship
    key = 'ship' if 'ship' in traj_dict else next(iter(traj_dict.keys()))
    traj = traj_dict[key]
    if torch.is_tensor(traj):
        traj = traj.cpu().numpy()
    meta = v.get('metadata', {})
    return traj.astype(np.float32), key, meta


def compute_displacements(traj: np.ndarray,
                          project_by_heading: bool = False) -> np.ndarray:
    """
    从代表轨迹 [K, S, 3] 计算每个 token 的位移向量 [K, 2]。
    - 默认：直接使用局部坐标的 (dx, dy)（常见 Δy≈0，因已对齐到首帧朝向）
    - project_by_heading=True：用末帧航向 θ 将位移模长 |Δr| 投影到全局方向
    """
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise ValueError('traj 形状应为 [K, S, 3]')
    disp_vec = traj[:, -1, :2] - traj[:, 0, :2]
    if not project_by_heading:
        return disp_vec.astype(np.float32)
    # 使用末帧航向，将位移模长映射到 (cosθ, sinθ)
    mag = np.linalg.norm(disp_vec, axis=1)
    theta = traj[:, -1, 2]
    dx = mag * np.cos(theta)
    dy = mag * np.sin(theta)
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def plot_vectors(vec: np.ndarray,
                 output: str,
                 origin=(0.0, 0.0),
                 alpha: float = 0.35,
                 width: float = 0.003,
                 clip_percentile: float = 99.5):
    if vec.ndim != 2 or vec.shape[1] != 2:
        raise ValueError('vec 形状应为 [K, 2]')
    dx, dy = vec[:, 0], vec[:, 1]
    ang = np.arctan2(dy, dx)
    hue = (ang + np.pi) / (2 * np.pi)
    colors = plt.cm.hsv(hue)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    X0 = np.full_like(dx, fill_value=float(origin[0]))
    Y0 = np.full_like(dy, fill_value=float(origin[1]))
    ax.quiver(
        X0, Y0,
        dx, dy,
        color=colors,
        angles='xy', scale_units='xy', scale=1.0,
        width=width, alpha=alpha, pivot='tail',
        headlength=5.0, headwidth=3.5, headaxislength=4.5,
    )

    # 自适应视窗（按分位数裁剪，避免极端值撑爆坐标轴）
    rx = np.percentile(np.abs(dx), clip_percentile)
    ry = np.percentile(np.abs(dy), clip_percentile)
    r = float(max(rx, ry))
    r = 1.05 * r if r > 0 else 1.0
    ax.set_xlim(origin[0] - r, origin[0] + r)
    ax.set_ylim(origin[1] - r, origin[1] + r)

    # 样式
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(origin[1], color='k', lw=0.6, alpha=0.3)
    ax.axvline(origin[0], color='k', lw=0.6, alpha=0.3)
    ax.scatter([origin[0]], [origin[1]], c='k', s=12, zorder=5)
    ax.set_xlabel('Δx (m)')
    ax.set_ylabel('Δy (m)')
    ax.set_title(f'Token vocabulary vectors (K={vec.shape[0]})')

    # 角度色条（-pi..pi）
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('angle (rad)')

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"✅ 已保存: {output}")


def main():
    ap = argparse.ArgumentParser(description='可视化词典为位移向量图')
    ap.add_argument('--vocab', type=str, required=True,
                    help='词典文件路径（如 smart/tokens/maritime_tokens_no_norm.pkl）')
    ap.add_argument('--output', type=str, default='assets/token_vocab.png',
                    help='输出图片路径')
    ap.add_argument('--origin_x', type=float, default=0.0, help='公共起点 X')
    ap.add_argument('--origin_y', type=float, default=0.0, help='公共起点 Y')
    ap.add_argument('--clip_percentile', type=float, default=99.5, help='视窗裁剪分位数')
    ap.add_argument('--project_by_heading', type=int, default=0,
                    help='1=按末帧航向投影 (dx,dy) 显示方向分布；0=直接用局部 (dx,dy)')
    args = ap.parse_args()

    traj, key, meta = load_vocab(args.vocab)
    vec = compute_displacements(traj, project_by_heading=bool(args.project_by_heading))
    print(f"词典键: {key}, token数: {vec.shape[0]}, 步数: {traj.shape[1]}")
    plot_vectors(
        vec,
        output=args.output,
        origin=(args.origin_x, args.origin_y),
        clip_percentile=args.clip_percentile,
    )


if __name__ == '__main__':
    main()


