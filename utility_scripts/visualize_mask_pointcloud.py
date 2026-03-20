# -*- coding: utf-8 -*-
"""
可视化带mask标记的点云

使用方法：
    python visualize_mask_pointcloud.py --frame 33
"""

import open3d as o3d
import numpy as np
import argparse
import os


def visualize_point_clouds(frame_id: int, data_dir: str = "data/mask_marked"):
    """
    可视化三种点云：
    1. 原始点云（灰色）
    2. 带标记的点云（mask区域为红色）
    3. 仅mask区域的点云（红色）
    """
    # 构建文件路径
    original_path = os.path.join(data_dir, f"frame_{frame_id:06d}_original.ply")
    marked_path = os.path.join(data_dir, f"frame_{frame_id:06d}_marked.ply")
    mask_only_path = os.path.join(data_dir, f"frame_{frame_id:06d}_mask_only.ply")
    
    print("="*70)
    print(f"Visualizing Point Clouds - Frame {frame_id}")
    print("="*70)
    
    # 检查文件是否存在
    for path in [original_path, marked_path, mask_only_path]:
        if not os.path.exists(path):
            print(f"[Error] File not found: {path}")
            return
    
    # 读取点云
    print("\n[Info] Loading point clouds...")
    
    original_pcd = o3d.io.read_point_cloud(original_path)
    marked_pcd = o3d.io.read_point_cloud(marked_path)
    mask_only_pcd = o3d.io.read_point_cloud(mask_only_path)
    
    print(f"[Info] Original point cloud: {len(original_pcd.points)} points")
    print(f"[Info] Marked point cloud: {len(marked_pcd.points)} points")
    print(f"[Info] Mask-only point cloud: {len(mask_only_pcd.points)} points")
    
    # 统计标记的点数
    if marked_pcd.has_colors():
        colors = np.asarray(marked_pcd.colors)
        # 红色点的阈值
        red_mask = (colors[:, 0] > 0.8) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
        red_count = np.sum(red_mask)
        print(f"[Info] Red marked points: {red_count}")
    
    # 创建可视化窗口
    print("\n[Info] Creating visualization...")
    
    # 创建三个可视化窗口
    vis_original = o3d.visualization.Visualizer()
    vis_original.create_window(window_name=f"Frame {frame_id} - Original", width=800, height=600)
    vis_original.add_geometry(original_pcd)
    render_opt = vis_original.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    
    vis_marked = o3d.visualization.Visualizer()
    vis_marked.create_window(window_name=f"Frame {frame_id} - Marked", width=800, height=600)
    vis_marked.add_geometry(marked_pcd)
    render_opt = vis_marked.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    
    vis_mask_only = o3d.visualization.Visualizer()
    vis_mask_only.create_window(window_name=f"Frame {frame_id} - Mask Only", width=800, height=600)
    vis_mask_only.add_geometry(mask_only_pcd)
    render_opt = vis_mask_only.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    
    print("[Info] Press 'q' or close window to exit")
    
    # 运行可视化
    while True:
        if not vis_original.poll_events() or not vis_marked.poll_events() or not vis_mask_only.poll_events():
            break
        vis_original.update_renderer()
        vis_marked.update_renderer()
        vis_mask_only.update_renderer()
    
    vis_original.destroy_window()
    vis_marked.destroy_window()
    vis_mask_only.destroy_window()
    
    print("[Info] Visualization closed")


def main():
    parser = argparse.ArgumentParser(description='Visualize marked point clouds')
    parser.add_argument('--frame', type=int, required=True, help='Frame ID to visualize')
    parser.add_argument('--data-dir', type=str, default='data/mask_marked', help='Data directory')
    
    args = parser.parse_args()
    
    visualize_point_clouds(args.frame, args.data_dir)


if __name__ == "__main__":
    main()
