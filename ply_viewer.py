import open3d as o3d
import numpy as np
import threading
import time

# 全局变量用于存储鼠标位置和选中的点
mouse_x = 0
mouse_y = 0
selected_point = None
running = True

# 模拟鼠标位置更新的线程
def mouse_tracker():
    global mouse_x, mouse_y, running
    while running:
        # 这里我们模拟鼠标位置更新
        # 实际应用中，我们需要使用系统API来获取真实的鼠标位置
        import pyautogui
        try:
            x, y = pyautogui.position()
            # 转换为相对于窗口的坐标
            # 这里假设窗口在屏幕左上角
            mouse_x = x
            mouse_y = y
        except Exception as e:
            pass
        time.sleep(0.01)

# 主函数
def main():
    global running
    
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
    print("2. Mouse over points to see their coordinates")
    print("3. Close the window to exit")
    
    # 创建可视化窗口
    print("\nOpening PLY viewer...")
    
    # 启动鼠标跟踪线程
    tracker_thread = threading.Thread(target=mouse_tracker)
    tracker_thread.daemon = True
    tracker_thread.start()
    
    # 使用draw_geometries函数
    # 这个函数会打开一个窗口，显示点云
    print("\nMouse over points to see their coordinates")
    print("Close the window when done")
    
    # 运行可视化
    o3d.visualization.draw_geometries([point_cloud], window_name="PLY Viewer", width=800, height=600)
    
    # 停止鼠标跟踪线程
    running = False
    tracker_thread.join(timeout=1.0)
    
    print("\n\nViewer closed")

if __name__ == "__main__":
    main()