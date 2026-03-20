"""
PLY点云文件可视化和交互工具

流程：
1. 读取PLY格式的点云文件
2. 检查点云的基本属性（是否有颜色和法向量）
3. 创建3*3的窗口布局
4. 计算每个窗口中点云数量和第一个点云的xyz
5. 增加框选功能，可以在窗口框选点云
6. 关闭窗口时结束程序
"""
import open3d as o3d
import numpy as np

# 主函数
def main():
    import os
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    # 读取PLY文件
    ply_file = "data/point_cloud_gen_high_quality.ply"
    print(f"Reading point cloud from {ply_file}...")
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(f"Point cloud has {len(point_cloud.points)} points")
    
    # 检查点云是否有颜色
    if point_cloud.has_colors():
        print("Point cloud has colors")
    else:
        print("Point cloud does not have colors")
    
    # 检查点云是否有法向量
    if point_cloud.has_normals():
        print("Point cloud has normals")
    else:
        print("Point cloud does not have normals")
    
    # 计算点云的边界框
    points = np.asarray(point_cloud.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    print(f"Point cloud bounds: min={min_bound}, max={max_bound}")
    
    # 将空间分成3*3的网格
    grid_size = 3
    grid_bounds = []
    for i in range(grid_size):
        for j in range(grid_size):
            x_min = min_bound[0] + (max_bound[0] - min_bound[0]) * i / grid_size
            x_max = min_bound[0] + (max_bound[0] - min_bound[0]) * (i + 1) / grid_size
            y_min = min_bound[1] + (max_bound[1] - min_bound[1]) * j / grid_size
            y_max = min_bound[1] + (max_bound[1] - min_bound[1]) * (j + 1) / grid_size
            grid_bounds.append((x_min, x_max, y_min, y_max))
    
    # 统计每个网格中的点云数量和第一个点的坐标
    grid_points = []
    has_output = False
    for i, (x_min, x_max, y_min, y_max) in enumerate(grid_bounds):
        # 筛选当前网格内的点
        grid_mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        grid_point_indices = np.where(grid_mask)[0]
        grid_point_count = len(grid_point_indices)
        
        if grid_point_count > 0:
            first_point = points[grid_point_indices[0]]
            print(f"Grid {i//3+1},{i%3+1}: {grid_point_count} points, first point: X={first_point[0]:.3f}, Y={first_point[1]:.3f}, Z={first_point[2]:.3f}")
            grid_points.append((grid_point_indices, (x_min, x_max, y_min, y_max)))
            has_output = True
    
    # 如果没有输出就退出
    if not has_output:
        print("No points found in any grid. Exiting.")
        return
    
    # 打印使用说明
    print("\nPLY Viewer Instructions:")
    print("1. Use the mouse to navigate the point cloud")
    print("2. Left-click and drag to select points")
    print("3. Selected points will be highlighted")
    print("4. Close the window to exit")
    
    # 创建可视化窗口
    print("\nOpening PLY viewer...")
    
    # 使用VisualizerWithEditing类，它支持框选功能
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="PLY Viewer with Grid and Selection", width=1200, height=900)
    vis.add_geometry(point_cloud)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    
    # 运行可视化
    print("\nLeft-click and drag to select points")
    print("Close the window when done")
    
    # 运行可视化
    vis.run()
    
    # 获取选中的点
    picked_points = vis.get_picked_points()
    
    # 销毁窗口
    vis.destroy_window()
    
    # 打印选中点的坐标
    if picked_points and len(picked_points) > 0:
        print("\n\nSelected points:")
        for i, point_idx in enumerate(picked_points[:10]):  # 只显示前10个点
            point = np.asarray(point_cloud.points)[point_idx]
            print(f"Point {i+1}: X: {point[0]:.3f}, Y: {point[1]:.3f}, Z: {point[2]:.3f}")
        if len(picked_points) > 10:
            print(f"... and {len(picked_points) - 10} more points")
    else:
        print("\n\nNo points selected")
    
    print("Viewer closed")

if __name__ == "__main__":
    main()