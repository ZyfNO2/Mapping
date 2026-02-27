import open3d as o3d
import numpy as np

# 主函数
def main():
    # 读取PLY文件
    ply_file = "point_cloud_gen.ply"
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
    
    # 打印使用说明
    print("\nPLY Viewer Instructions:")
    print("1. Use the mouse to navigate the point cloud")
    print("2. Left-click to select points")
    print("3. Selected points will be displayed in the console")
    print("4. Close the window to exit")
    
    # 创建可视化窗口
    print("\nOpening PLY viewer...")
    
    # 使用VisualizerWithEditing类
    # 这个类允许用户选择点并显示其坐标
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="PLY Viewer with Selection", width=800, height=600)
    vis.add_geometry(point_cloud)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    
    # 运行可视化
    print("\nLeft-click to select points, right-click to cancel")
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
        for i, point_idx in enumerate(picked_points):
            point = np.asarray(point_cloud.points)[point_idx]
            print(f"Point {i+1}: X: {point[0]:.3f}, Y: {point[1]:.3f}, Z: {point[2]:.3f}")
    else:
        print("\n\nNo points selected")
    
    print("Viewer closed")

if __name__ == "__main__":
    main()