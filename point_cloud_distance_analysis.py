"""
点云相邻点距离分析工具

功能：
1. 读取PLY格式的点云文件
2. 构建KDTree以快速查找最近邻点
3. 计算每个点与其最近邻点的距离
4. 分析距离的统计信息（平均值、最小值、最大值、标准差）
5. 生成距离分布直方图
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Point Cloud Neighbor Distance Analysis")
    parser.add_argument("input_file", type=str, help="Input PLY file path")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors to consider")
    parser.add_argument("--visualize", action="store_true", help="Visualize the distance distribution")
    
    args = parser.parse_args()
    
    # 读取点云文件
    print(f"Reading point cloud from {args.input_file}...")
    point_cloud = o3d.io.read_point_cloud(args.input_file)
    print(f"Point cloud has {len(point_cloud.points)} points")
    
    # 检查点云是否为空
    if len(point_cloud.points) == 0:
        print("Error: Empty point cloud")
        return
    
    # 转换为numpy数组
    points = np.asarray(point_cloud.points)
    
    # 构建KDTree
    print("Building KDTree for nearest neighbor search...")
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # 计算每个点与其最近邻点的距离
    print("Calculating nearest neighbor distances...")
    distances = []
    for i, point in enumerate(points):
        # 查找最近邻点（k+1因为第一个是点本身）
        [_, idx, dist] = kdtree.search_knn_vector_3d(point, args.k + 1)
        # 忽略点本身，只取其他最近邻点的距离
        for d in dist[1:]:
            distances.append(np.sqrt(d))  # dist返回的是距离的平方
        
        # 显示进度
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(points)} points")
    
    # 转换为numpy数组以便统计分析
    distances = np.array(distances)
    
    # 计算统计信息
    print("\nDistance Statistics:")
    print(f"Mean distance: {np.mean(distances):.6f} meters")
    print(f"Median distance: {np.median(distances):.6f} meters")
    print(f"Minimum distance: {np.min(distances):.6f} meters")
    print(f"Maximum distance: {np.max(distances):.6f} meters")
    print(f"Standard deviation: {np.std(distances):.6f} meters")
    print(f"25th percentile: {np.percentile(distances, 25):.6f} meters")
    print(f"75th percentile: {np.percentile(distances, 75):.6f} meters")
    
    # 可视化距离分布
    if args.visualize:
        print("\nGenerating distance distribution histogram...")
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=100, alpha=0.75)
        plt.title('Nearest Neighbor Distance Distribution')
        plt.xlabel('Distance (meters)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('distance_distribution.png')
        print("Distance distribution histogram saved as 'distance_distribution.png'")
    
    # 保存距离数据到文件
    np.savetxt('neighbor_distances.txt', distances)
    print("\nNeighbor distances saved to 'neighbor_distances.txt'")

if __name__ == "__main__":
    main()
