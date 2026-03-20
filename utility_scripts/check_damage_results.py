# -*- coding: utf-8 -*-
"""
检查损伤检测结果
"""

import open3d as o3d
import numpy as np
import os

# 设置路径
svo_name = "HD2K_SN36245620_15-17-12"
data_dir = f"data/damage_detection/{svo_name}"

print("="*70)
print(f"Damage Detection Results - {svo_name}")
print("="*70)

# 读取三种点云
original_path = os.path.join(data_dir, f"{svo_name}_original.ply")
marked_path = os.path.join(data_dir, f"{svo_name}_marked.ply")
damage_only_path = os.path.join(data_dir, f"{svo_name}_damage_only.ply")

print("\nLoading point clouds...")
original_pcd = o3d.io.read_point_cloud(original_path)
marked_pcd = o3d.io.read_point_cloud(marked_path)
damage_only_pcd = o3d.io.read_point_cloud(damage_only_path)

# 获取统计信息
original_points = np.asarray(original_pcd.points)
marked_points = np.asarray(marked_pcd.points)
damage_only_points = np.asarray(damage_only_pcd.points)

print(f"\nPoint Cloud Statistics:")
print(f"  Original point cloud:  {len(original_points):>8,} points")
print(f"  Marked point cloud:    {len(marked_points):>8,} points")
print(f"  Damage-only point cloud: {len(damage_only_points):>8,} points")

# 统计红色标记点
if marked_pcd.has_colors():
    marked_colors = np.asarray(marked_pcd.colors)
    red_mask = (marked_colors[:, 0] > 0.8) & (marked_colors[:, 1] < 0.3) & (marked_colors[:, 2] < 0.3)
    red_count = np.sum(red_mask)
    print(f"\n  Red marked points:     {red_count:>8,} points ({red_count/len(marked_points)*100:.2f}%)")

# 显示边界框
print(f"\nPoint Cloud Bounds:")
print(f"  X: {original_points[:,0].min():.3f} to {original_points[:,0].max():.3f}")
print(f"  Y: {original_points[:,1].min():.3f} to {original_points[:,1].max():.3f}")
print(f"  Z: {original_points[:,2].min():.3f} to {original_points[:,2].max():.3f}")

# 显示损伤区域样本点
if len(damage_only_points) > 0:
    print(f"\nSample Damage Points (first 5):")
    for i in range(min(5, len(damage_only_points))):
        print(f"  {damage_only_points[i]}")

# 文件大小
print(f"\nFile Sizes:")
for name in ["original", "marked", "damage_only"]:
    path = os.path.join(data_dir, f"{svo_name}_{name}.ply")
    size = os.path.getsize(path) / (1024 * 1024)  # MB
    print(f"  {svo_name}_{name}.ply: {size:.2f} MB")

print("\n" + "="*70)
print("All files generated successfully!")
print("="*70)
