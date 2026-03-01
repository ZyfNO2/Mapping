"""
文件目的：点云文件读取和格式转换工具

处理流程：
1. 尝试按顺序读取不同格式的点云文件（PLY → OBJ → CSV）
2. 对不同格式的文件进行相应的解析和处理
3. 预览点云数据
4. 将点云保存为PCD格式

数据处理方式：
- PLY文件：直接使用Open3D读取
- OBJ文件：读取网格并通过泊松圆盘采样转换为点云
- CSV文件：根据不同的CSV格式解析点坐标和颜色信息，支持多种颜色存储格式
"""
import open3d as o3d
import numpy as np
import csv

# 读取CSV文件
def read_csv_point_cloud(csv_file):
    points = []
    colors = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 读取表头
        
        # 检查CSV文件格式
        if len(header) == 4 and header[3] == 'rgba':
            # 格式：x, y, z, rgba
            for i, row in enumerate(reader):
                x, y, z, rgba = map(float, row)
                points.append([x, y, z])
                
                # 处理rgba值
                # 尝试将rgba转换为整数，然后提取RGB值
                try:
                    # 确保rgba是一个整数
                    rgba_int = int(rgba)
                    # 尝试ARGB布局解码 (从高位到低位：Alpha, Red, Green, Blue)
                    a = ((rgba_int >> 24) & 0xFF)
                    r = ((rgba_int >> 16) & 0xFF) / 255.0
                    g = ((rgba_int >> 8) & 0xFF) / 255.0
                    b = (rgba_int & 0xFF) / 255.0
                except:
                    # 如果处理失败，使用基于坐标的颜色
                    r = (abs(x) % 1.0)
                    g = (abs(y) % 1.0)
                    b = (abs(z) % 1.0)
                
                # 确保颜色值在0-1范围内
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                
                colors.append([r, g, b])
        elif len(header) == 7 and header[3] == 'r' and header[4] == 'g' and header[5] == 'b' and header[6] == 'a':
            # 格式：x, y, z, r, g, b, a
            for i, row in enumerate(reader):
                x, y, z, r, g, b, a = map(float, row)
                points.append([x, y, z])
                # 确保颜色值在0-1范围内
                r_int = int(max(0, min(255, r)))
                g_int = int(max(0, min(255, g)))
                b_int = int(max(0, min(255, b)))
                # 归一化到0-1范围
                colors.append([r_int/255.0, g_int/255.0, b_int/255.0])
        elif len(header) == 12 and header[3] == 'rgba' and header[4] == 'r_rgba' and header[5] == 'g_rgba' and header[6] == 'b_rgba' and header[7] == 'a_rgba' and header[8] == 'r_argb' and header[9] == 'g_argb' and header[10] == 'b_argb' and header[11] == 'a_argb':
            # 格式：x, y, z, rgba, r_rgba, g_rgba, b_rgba, a_rgba, r_argb, g_argb, b_argb, a_argb
            for i, row in enumerate(reader):
                x, y, z, rgba, r_rgba, g_rgba, b_rgba, a_rgba, r_argb, g_argb, b_argb, a_argb = map(float, row)
                points.append([x, y, z])
                # 使用ARGB布局的颜色值
                r = float(r_argb) / 255.0
                g = float(g_argb) / 255.0
                b = float(b_argb) / 255.0
                # 确保颜色值在0-1范围内
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                colors.append([r, g, b])
        elif len(header) == 8 and header[3] == 'rgba' and header[4] == 'r' and header[5] == 'g' and header[6] == 'b' and header[7] == 'a':
            # 格式：x, y, z, rgba, r, g, b, a
            for i, row in enumerate(reader):
                x, y, z, rgba, r, g, b, a = map(float, row)
                points.append([x, y, z])
                # 确保颜色值在0-1范围内
                r_int = int(max(0, min(255, r)))
                g_int = int(max(0, min(255, g)))
                b_int = int(max(0, min(255, b)))
                # 归一化到0-1范围
                colors.append([r_int/255.0, g_int/255.0, b_int/255.0])
        else:
            # 格式：x, y, z, r, g, b
            for i, row in enumerate(reader):
                x, y, z, r, g, b = map(float, row)
                points.append([x, y, z])
                # 确保颜色值在0-1范围内
                r_int = int(max(0, min(255, r)))
                g_int = int(max(0, min(255, g)))
                b_int = int(max(0, min(255, b)))
                # 归一化到0-1范围
                colors.append([r_int/255.0, g_int/255.0, b_int/255.0])
    
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return point_cloud

# 读取OBJ文件
def read_obj_point_cloud(obj_file):
    # 使用open3d直接读取OBJ文件
    mesh = o3d.io.read_triangle_mesh(obj_file)
    # 将网格转换为点云
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=100000)
    return point_cloud

# 主函数
def main():
    import os
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    obj_file = "data/point_cloud_gen.obj"
    csv_file = "data/point_cloud_gen_argb.csv"
    ply_file = "data/point_cloud_gen.ply"
    pcd_file = "data/point_cloud_gen.pcd"
    
    try:
        # 尝试读取PLY文件
        print(f"Reading point cloud from {ply_file}...")
        point_cloud = o3d.io.read_point_cloud(ply_file)
        print(f"Point cloud has {len(point_cloud.points)} points")
        if point_cloud.has_colors():
            print("Point cloud has colors")
        else:
            print("Point cloud does not have colors")
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        # 如果PLY文件读取失败，尝试读取OBJ文件
        try:
            print(f"Reading point cloud from {obj_file}...")
            point_cloud = read_obj_point_cloud(obj_file)
            print(f"Point cloud has {len(point_cloud.points)} points")
            if point_cloud.has_colors():
                print("Point cloud has colors")
            else:
                print("Point cloud does not have colors")
        except Exception as e:
            print(f"Error reading OBJ file: {e}")
            # 如果OBJ文件读取失败，尝试读取CSV文件
            print(f"Reading point cloud from {csv_file}...")
            point_cloud = read_csv_point_cloud(csv_file)
            print(f"Point cloud has {len(point_cloud.points)} points")
    
    # 预览点云
    print("Previewing point cloud...")
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Preview")
    
    # 保存为PCD格式
    print(f"Saving point cloud to {pcd_file}...")
    o3d.io.write_point_cloud(pcd_file, point_cloud)
    print(f"Point cloud saved as PCD: {pcd_file}")

if __name__ == "__main__":
    main()