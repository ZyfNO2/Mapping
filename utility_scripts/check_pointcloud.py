# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np

# 读取原始点云
pcd = o3d.io.read_point_cloud('data/mask_marked/frame_000033_original.ply')
print(f'Original point cloud has {len(pcd.points)} points')

if len(pcd.points) > 0:
    points = np.asarray(pcd.points)
    print(f'Sample points:')
    for i in range(min(5, len(points))):
        print(f'  {points[i]}')
    print(f'Point cloud bounds:')
    print(f'  X: {points[:,0].min():.3f} to {points[:,0].max():.3f}')
    print(f'  Y: {points[:,1].min():.3f} to {points[:,1].max():.3f}')
    print(f'  Z: {points[:,2].min():.3f} to {points[:,2].max():.3f}')

# 读取mask-only点云
mask_pcd = o3d.io.read_point_cloud('data/mask_marked/frame_000033_mask_only.ply')
print(f'\nMask-only point cloud has {len(mask_pcd.points)} points')

if len(mask_pcd.points) > 0:
    points = np.asarray(mask_pcd.points)
    print(f'Sample mask points:')
    for i in range(min(5, len(points))):
        print(f'  {points[i]}')
