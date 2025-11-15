#!/usr/bin/env python3
"""
Maritime Token Visualization (English Version)
Generate three analysis plots:
1. All trajectory vectors (from origin)
2. Displacement distribution
3. Direction distribution
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set font
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_tokens(token_path):
    """Load token file"""
    print(f"Loading token file: {token_path}")
    with open(token_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract ship trajectory data
    traj = data['traj']['ship']  # (N, 2, 3) - N tokens, 2 time points, (x, y, theta)
    
    print(f"Loaded successfully")
    print(f"   Token count: {traj.shape[0]}")
    print(f"   Time steps: {traj.shape[1]}")
    print(f"   Feature dimensions: {traj.shape[2]} (x, y, theta)")
    
    return traj, data


def visualize_all_vectors(traj, output_path):
    """
    Plot 1: All vectors from origin
    Show all token motion vectors
    """
    print(f"\nGenerating Plot 1: All trajectory vectors...")
    
    # Calculate start to end vectors
    start_points = traj[:, 0, :2]  # (N, 2) - start point (x, y)
    end_points = traj[:, 1, :2]    # (N, 2) - end point (x, y)
    vectors = end_points - start_points  # (N, 2) - displacement vector
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw all vectors from origin
    origin = np.zeros((len(vectors), 2))
    
    # Use color map to represent vector length
    vector_lengths = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # Draw vectors
    quiver = ax.quiver(origin[:, 0], origin[:, 1], 
                       vectors[:, 0], vectors[:, 1],
                       vector_lengths,
                       cmap='viridis',
                       alpha=0.6,
                       scale_units='xy',
                       scale=1,
                       width=0.003)
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('Displacement (m)', fontsize=12)
    
    # Set axes
    max_range = max(np.abs(vectors).max(), 150)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Add direction indicators
    ax.text(max_range * 0.9, 0, 'East', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(-max_range * 0.9, 0, 'West', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(0, max_range * 0.9, 'North', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(0, -max_range * 0.9, 'South', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Title and labels
    ax.set_title(f'Maritime Token Trajectory Vectors ({len(vectors)} Tokens)\nAll vectors start from origin', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Displacement (m)', fontsize=14)
    ax.set_ylabel('Y Displacement (m)', fontsize=14)
    
    # Add statistics
    info_text = f'Mean displacement: {vector_lengths.mean():.1f}m\n'
    info_text += f'Max displacement: {vector_lengths.max():.1f}m\n'
    info_text += f'Min displacement: {vector_lengths.min():.1f}m'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_displacement_distribution(traj, output_path):
    """
    Plot 2: Displacement distribution
    """
    print(f"\nGenerating Plot 2: Displacement distribution...")
    
    # Calculate displacement
    start_points = traj[:, 0, :2]
    end_points = traj[:, 1, :2]
    vectors = end_points - start_points
    displacements = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Histogram
    n, bins, patches = ax1.hist(displacements, bins=50, 
                                  color='steelblue', 
                                  edgecolor='black', 
                                  alpha=0.7)
    
    # Color by height
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax1.axvline(displacements.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {displacements.mean():.1f}m')
    ax1.axvline(np.median(displacements), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(displacements):.1f}m')
    ax1.set_xlabel('Displacement (m)', fontsize=14)
    ax1.set_ylabel('Token Count', fontsize=14)
    ax1.set_title('Displacement Distribution Histogram', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Cumulative distribution
    sorted_disp = np.sort(displacements)
    cumulative = np.arange(1, len(sorted_disp) + 1) / len(sorted_disp) * 100
    
    ax2.plot(sorted_disp, cumulative, linewidth=2, color='steelblue')
    ax2.axhline(50, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Mark key percentiles
    p50 = np.percentile(displacements, 50)
    p95 = np.percentile(displacements, 95)
    ax2.axvline(p50, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(p95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(p50, 55, f'P50: {p50:.1f}m', fontsize=10, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.text(p95, 97, f'P95: {p95:.1f}m', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlabel('Displacement (m)', fontsize=14)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=14)
    ax2.set_title('Cumulative Distribution Curve', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Statistics:\n'
    stats_text += f'Sample count: {len(displacements)}\n'
    stats_text += f'Mean: {displacements.mean():.2f} m\n'
    stats_text += f'Std dev: {displacements.std():.2f} m\n'
    stats_text += f'Min: {displacements.min():.2f} m\n'
    stats_text += f'Max: {displacements.max():.2f} m\n'
    stats_text += f'P25: {np.percentile(displacements, 25):.2f} m\n'
    stats_text += f'P50: {np.percentile(displacements, 50):.2f} m\n'
    stats_text += f'P75: {np.percentile(displacements, 75):.2f} m\n'
    stats_text += f'P95: {np.percentile(displacements, 95):.2f} m'
    
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Saved: {output_path}")
    plt.close()


def visualize_direction_distribution(traj, output_path):
    """
    Plot 3: Direction distribution
    Including polar plot and quadrant statistics
    """
    print(f"\nGenerating Plot 3: Direction distribution...")
    
    # Calculate direction
    start_points = traj[:, 0, :2]
    end_points = traj[:, 1, :2]
    vectors = end_points - start_points
    directions = np.arctan2(vectors[:, 1], vectors[:, 0])  # radians, range [-pi, pi]
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Left plot: Polar histogram
    ax1 = plt.subplot(131, projection='polar')
    
    # Convert direction to [0, 2pi]
    directions_positive = directions.copy()
    directions_positive[directions_positive < 0] += 2 * np.pi
    
    # Draw polar histogram
    n_bins = 36  # One bin per 10 degrees
    theta_bins = np.linspace(0, 2*np.pi, n_bins + 1)
    radii, _ = np.histogram(directions_positive, bins=theta_bins)
    theta = (theta_bins[:-1] + theta_bins[1:]) / 2
    width = 2 * np.pi / n_bins
    
    bars = ax1.bar(theta, radii, width=width, bottom=0, alpha=0.7)
    
    # Color by height
    cm = plt.cm.viridis
    for r, bar in zip(radii, bars):
        bar.set_facecolor(cm(r / radii.max()))
        bar.set_edgecolor('black')
    
    ax1.set_theta_zero_location('E')  # 0 degrees at East (right)
    ax1.set_theta_direction(1)  # Counter-clockwise
    ax1.set_title('Direction Distribution Polar Plot\n(0deg=East, 90deg=North)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True)
    
    # Middle plot: Direction histogram (degrees)
    ax2 = plt.subplot(132)
    
    # Convert to degrees
    directions_deg = np.degrees(directions)
    
    n, bins, patches = ax2.hist(directions_deg, bins=36, 
                                  range=(-180, 180),
                                  color='steelblue', 
                                  edgecolor='black', 
                                  alpha=0.7)
    
    # Color
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = n / n.max()
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Mark key directions
    ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='East(0deg)')
    ax2.axvline(90, color='green', linestyle='--', linewidth=1, alpha=0.7, label='North(90deg)')
    ax2.axvline(-90, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='South(-90deg)')
    ax2.axvline(180, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(-180, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='West(±180deg)')
    
    ax2.set_xlabel('Direction (degrees)', fontsize=14)
    ax2.set_ylabel('Token Count', fontsize=14)
    ax2.set_title('Direction Distribution Histogram', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Right plot: Quadrant statistics
    ax3 = plt.subplot(133)
    
    # Calculate count per quadrant
    Q1 = np.sum((directions >= 0) & (directions < np.pi/2))      # Northeast
    Q2 = np.sum((directions >= np.pi/2) & (directions <= np.pi)) # Northwest
    Q3 = np.sum((directions >= -np.pi) & (directions < -np.pi/2)) # Southwest
    Q4 = np.sum((directions >= -np.pi/2) & (directions < 0))     # Southeast
    
    quadrants = ['Q1\nNortheast\n(0-90deg)', 'Q2\nNorthwest\n(90-180deg)', 
                 'Q3\nSouthwest\n(-180--90deg)', 'Q4\nSoutheast\n(-90-0deg)']
    counts = [Q1, Q2, Q3, Q4]
    percentages = [c / len(directions) * 100 for c in counts]
    
    # Draw bar chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax3.bar(range(4), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add expected value reference line
    expected = len(directions) / 4
    ax3.axhline(expected, color='red', linestyle='--', linewidth=2, 
                label=f'Expected: {expected:.0f} (25%)')
    
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(quadrants, fontsize=11)
    ax3.set_ylabel('Token Count', fontsize=14)
    ax3.set_title('Quadrant Distribution Statistics', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Calculate chi-square test
    chi_square = sum((c - expected)**2 / expected for c in counts)
    
    # Add statistics
    stats_text = f'Chi-square: {chi_square:.2f}\n'
    stats_text += f'(< 50 indicates good uniformity)\n'
    if chi_square < 50:
        stats_text += 'Distribution is uniform'
    else:
        stats_text += 'Distribution is non-uniform'
    
    ax3.text(0.5, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function"""
    print("=" * 80)
    print("Maritime Token Vocabulary Visualization")
    print("=" * 80)
    
    # File paths
    token_path = Path("smart/tokens/maritime_tokens_no_norm.pkl")
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    traj, data = load_tokens(token_path)
    
    # Generate three plots
    visualize_all_vectors(traj, output_dir / "token_all_vectors_en.png")
    visualize_displacement_distribution(traj, output_dir / "token_displacement_dist_en.png")
    visualize_direction_distribution(traj, output_dir / "token_direction_dist_en.png")
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"\nGenerated images:")
    print(f"  1. {output_dir / 'token_all_vectors_en.png'}")
    print(f"  2. {output_dir / 'token_displacement_dist_en.png'}")
    print(f"  3. {output_dir / 'token_direction_dist_en.png'}")
    print()


if __name__ == "__main__":
    main()

