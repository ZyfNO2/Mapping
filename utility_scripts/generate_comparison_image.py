# -*- coding: utf-8 -*-
"""
生成点云对比图像
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def generate_comparison(frame_id: int, data_dir: str = "data/mask_marked"):
    """生成点云对比图"""
    
    # 读取点云
    original_path = os.path.join(data_dir, f"frame_{frame_id:06d}_original.ply")
    marked_path = os.path.join(data_dir, f"frame_{frame_id:06d}_marked.ply")
    mask_only_path = os.path.join(data_dir, f"frame_{frame_id:06d}_mask_only.ply")
    
    print("Loading point clouds...")
    original_pcd = o3d.io.read_point_cloud(original_path)
    marked_pcd = o3d.io.read_point_cloud(marked_path)
    mask_only_pcd = o3d.io.read_point_cloud(mask_only_path)
    
    # 获取点云数据
    original_points = np.asarray(original_pcd.points)
    marked_points = np.asarray(marked_pcd.points)
    mask_only_points = np.asarray(mask_only_pcd.points)
    
    if marked_pcd.has_colors():
        marked_colors = np.asarray(marked_pcd.colors)
    else:
        marked_colors = np.ones((len(marked_points), 3)) * 0.7
    
    # 创建图形
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1: 原始点云
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original_points[::10, 0], original_points[::10, 1], original_points[::10, 2], 
                c='gray', s=0.1, alpha=0.5)
    ax1.set_title(f'Original Point Cloud\n({len(original_points)} points)', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 子图2: 带标记的点云
    ax2 = fig.add_subplot(132, projection='3d')
    # 采样显示（太多点会卡顿）
    sample_idx = np.random.choice(len(marked_points), min(50000, len(marked_points)), replace=False)
    ax2.scatter(marked_points[sample_idx, 0], marked_points[sample_idx, 1], marked_points[sample_idx, 2],
                c=marked_colors[sample_idx], s=0.1, alpha=0.5)
    ax2.set_title(f'Marked Point Cloud\n({len(marked_points)} points, red=mask region)', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 子图3: 仅mask区域
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(mask_only_points[:, 0], mask_only_points[:, 1], mask_only_points[:, 2],
                c='red', s=1, alpha=0.8)
    ax3.set_title(f'Mask Only Point Cloud\n({len(mask_only_points)} points)', fontsize=12)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(data_dir, f"frame_{frame_id:06d}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison image to: {output_path}")
    
    # 显示统计信息
    print("\n" + "="*70)
    print(f"Point Cloud Statistics - Frame {frame_id}")
    print("="*70)
    print(f"Original point cloud:  {len(original_points):>8} points")
    print(f"Marked point cloud:    {len(marked_points):>8} points")
    print(f"Mask-only point cloud: {len(mask_only_points):>8} points")
    
    # 统计红色点
    red_mask = (marked_colors[:, 0] > 0.8) & (marked_colors[:, 1] < 0.3) & (marked_colors[:, 2] < 0.3)
    red_count = np.sum(red_mask)
    print(f"Red marked points:     {red_count:>8} points ({red_count/len(marked_points)*100:.2f}%)")
    
    print("\nBounds:")
    print(f"  X: {original_points[:,0].min():.3f} to {original_points[:,0].max():.3f}")
    print(f"  Y: {original_points[:,1].min():.3f} to {original_points[:,1].max():.3f}")
    print(f"  Z: {original_points[:,2].min():.3f} to {original_points[:,2].max():.3f}")
    
    plt.close()
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=33)
    parser.add_argument('--data-dir', type=str, default='data/mask_marked')
    args = parser.parse_args()
    
    generate_comparison(args.frame, args.data_dir)
