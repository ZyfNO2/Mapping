# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np

# 读取原始点云
original_pcd = o3d.io.read_point_cloud('data/mask_marked/frame_000033_original.ply')
original_points = np.asarray(original_pcd.points)
print(f'Original point cloud has {len(original_points)} points')

# 读取mask-only点云
mask_pcd = o3d.io.read_point_cloud('data/mask_marked/frame_000033_mask_only.ply')
mask_points = np.asarray(mask_pcd.points)
print(f'Mask-only point cloud has {len(mask_points)} points')

# 创建空间哈希集合
mask_points_set = set()
for pt in mask_points:
    key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
    mask_points_set.add(key)

print(f'Mask points set size: {len(mask_points_set)}')

# 检查有多少原始点云能匹配
matched_count = 0
sample_original_keys = []
for i, pt in enumerate(original_points):
    key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
    if i < 5:
        sample_original_keys.append(key)
    if key in mask_points_set:
        matched_count += 1

print(f'\nSample original point keys (quantized):')
for key in sample_original_keys:
    print(f'  {key}')

print(f'\nSample mask point keys (quantized):')
for pt in mask_points[:5]:
    key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
    print(f'  {key}')

print(f'\nMatched points: {matched_count}')

# 尝试更宽松的匹配（扩大搜索范围）
print('\nTrying relaxed matching with 1cm tolerance...')
matched_relaxed = 0
for pt in original_points:
    for mask_pt in mask_points:
        dist = np.linalg.norm(pt - mask_pt)
        if dist < 0.01:  # 1cm
            matched_relaxed += 1
            break

print(f'Matched with 1cm tolerance: {matched_relaxed}')

# 尝试2cm容忍度
print('\nTrying relaxed matching with 2cm tolerance...')
matched_relaxed2 = 0
for pt in original_points:
    for mask_pt in mask_points:
        dist = np.linalg.norm(pt - mask_pt)
        if dist < 0.02:  # 2cm
            matched_relaxed2 += 1
            break

print(f'Matched with 2cm tolerance: {matched_relaxed2}')
