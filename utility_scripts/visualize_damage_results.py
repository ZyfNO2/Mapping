# -*- coding: utf-8 -*-
"""
可视化损伤检测结果
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置路径
svo_name = "HD2K_SN36245620_15-17-12"
data_dir = f"data/damage_detection/{svo_name}"

print("Loading point clouds...")
original_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f"{svo_name}_original.ply"))
marked_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f"{svo_name}_marked.ply"))
damage_only_pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f"{svo_name}_damage_only.ply"))

# 获取数据
original_points = np.asarray(original_pcd.points)
marked_points = np.asarray(marked_pcd.points)
marked_colors = np.asarray(marked_pcd.colors)
damage_only_points = np.asarray(damage_only_pcd.points)

# 创建图形
fig = plt.figure(figsize=(18, 6))

# 子图1: 原始点云
ax1 = fig.add_subplot(131, projection='3d')
sample_idx1 = np.random.choice(len(original_points), min(30000, len(original_points)), replace=False)
ax1.scatter(original_points[sample_idx1, 0], original_points[sample_idx1, 1], original_points[sample_idx1, 2], 
            c='gray', s=0.1, alpha=0.5)
ax1.set_title(f'Original Point Cloud\n({len(original_points):,} points)', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 子图2: 带标记的点云
ax2 = fig.add_subplot(132, projection='3d')
# 采样显示
sample_idx2 = np.random.choice(len(marked_points), min(50000, len(marked_points)), replace=False)
ax2.scatter(marked_points[sample_idx2, 0], marked_points[sample_idx2, 1], marked_points[sample_idx2, 2],
            c=marked_colors[sample_idx2], s=0.1, alpha=0.5)
ax2.set_title(f'Marked Point Cloud\n({len(marked_points):,} points, red=damage)', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# 子图3: 仅损伤区域
ax3 = fig.add_subplot(133, projection='3d')
sample_idx3 = np.random.choice(len(damage_only_points), min(30000, len(damage_only_points)), replace=False)
ax3.scatter(damage_only_points[sample_idx3, 0], damage_only_points[sample_idx3, 1], damage_only_points[sample_idx3, 2],
            c='red', s=0.2, alpha=0.6)
ax3.set_title(f'Damage Only Point Cloud\n({len(damage_only_points):,} points)', fontsize=12)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()

# 保存图像
output_path = os.path.join(data_dir, f"{svo_name}_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison image to: {output_path}")

plt.close()

print("\nVisualization completed!")
