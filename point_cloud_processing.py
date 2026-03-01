"""
文件目的：点云数据处理和优化工具

处理流程：
1. 加载点云文件
2. 体素下采样减少点云数据量
3. 统计滤波或半径滤波去除离群点
4. 法线估计计算每个点的法线向量
5. 基于法线/曲率的异常点剔除
6. 聚类分离并移除小碎片
7. 保存处理后的点云

数据处理方式：
- 体素下采样：使用指定大小的体素网格对点云进行降采样
- 滤波：根据统计特性或局部密度去除离群点
- 法线估计：使用KDTree搜索近邻点计算法线
- 曲率计算：通过协方差矩阵特征值计算局部曲率
- 聚类：使用DBSCAN算法对点云进行聚类，移除小尺寸聚类
"""
import open3d as o3d
import numpy as np
import argparse

# 读取点云文件
def load_point_cloud(file_path):
    import os
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    print(f"Loading point cloud from {file_path}...")
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"Original point cloud has {len(point_cloud.points)} points")
    if point_cloud.has_colors():
        print("Point cloud has colors")
    else:
        print("Point cloud does not have colors")
    return point_cloud

# 密度过滤 - 去除离散点，保留稠密区域
def density_filter(point_cloud, radius=0.06, min_density=10):
    print(f"Density filtering with radius: {radius}, min density: {min_density}...")
    
    # 计算每个点的局部密度
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    dense_points = []
    dense_colors = []
    
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    
    for i, point in enumerate(points):
        # 搜索周围一定半径内的点
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        # 如果局部密度足够高，保留该点
        if len(idx) >= min_density:
            dense_points.append(point)
            if colors is not None:
                dense_colors.append(colors[i])
    
    # 创建新的点云
    dense_cloud = o3d.geometry.PointCloud()
    dense_cloud.points = o3d.utility.Vector3dVector(np.array(dense_points))
    if dense_colors:
        dense_cloud.colors = o3d.utility.Vector3dVector(np.array(dense_colors))
    
    print(f"Density filtered point cloud has {len(dense_cloud.points)} points")
    return dense_cloud

# 统计滤波
def statistical_outlier_removal(point_cloud, nb_neighbors=20, std_ratio=2.0):
    print(f"Statistical outlier removal with {nb_neighbors} neighbors and {std_ratio} std ratio...")
    filtered, indices = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Filtered point cloud has {len(filtered.points)} points")
    return filtered

# 半径滤波
def radius_outlier_removal(point_cloud, nb_points=16, radius=0.1):
    print(f"Radius outlier removal with {nb_points} points and {radius} radius...")
    filtered, indices = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    print(f"Filtered point cloud has {len(filtered.points)} points")
    return filtered

# 法线估计
def estimate_normals(point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    print("Estimating normals...")
    point_cloud.estimate_normals(search_param=search_param)
    print("Normals estimated successfully")
    return point_cloud

# 基于法线/曲率的异常点剔除
def remove_outliers_based_on_normals(point_cloud, threshold=0.5):
    print("Removing outliers based on normals/curvature...")
    # 计算曲率
    point_cloud.estimate_normals()
    curvatures = []
    
    # 只创建一次KDTree以提高效率
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    for i in range(len(point_cloud.points)):
        # 计算局部曲率
        [_, idx, _] = kdtree.search_knn_vector_3d(point_cloud.points[i], 10)
        # 确保有足够的邻居
        if len(idx) > 1:
            neighbors = np.asarray(point_cloud.points)[idx[1:], :]
            center = np.asarray(point_cloud.points)[idx[0], :]
            # 计算协方差矩阵
            cov = np.cov(neighbors - center, rowvar=False)
            # 计算特征值
            eigenvalues, _ = np.linalg.eigh(cov)
            # 计算曲率
            curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-10)
            curvatures.append(curvature)
        else:
            # 如果邻居不足，设为高曲率（视为异常点）
            curvatures.append(1.0)
    
    # 根据曲率阈值过滤
    curvatures = np.array(curvatures)
    mask = curvatures < threshold
    filtered_points = np.asarray(point_cloud.points)[mask]
    filtered_colors = np.asarray(point_cloud.colors)[mask] if point_cloud.has_colors() else None
    
    # 创建新的点云
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"Filtered point cloud has {len(filtered_cloud.points)} points")
    return filtered_cloud

# 聚类分离小碎片
def cluster_and_remove_small_clusters(point_cloud, eps=0.1, min_points=50):
    print(f"Clustering with eps={eps} and min_points={min_points}...")
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")
    
    # 统计每个聚类的大小
    cluster_sizes = np.bincount(labels[labels >= 0])
    print(f"Cluster sizes: {cluster_sizes}")
    
    # 只保留大聚类
    large_clusters = np.where(cluster_sizes >= min_points)[0]
    print(f"Large clusters: {large_clusters}")
    
    # 如果没有找到大聚类，返回原始点云
    if len(large_clusters) == 0:
        print("No large clusters found, returning original point cloud")
        return point_cloud
    
    # 过滤点云
    mask = np.isin(labels, large_clusters)
    filtered_points = np.asarray(point_cloud.points)[mask]
    filtered_colors = np.asarray(point_cloud.colors)[mask] if point_cloud.has_colors() else None
    
    # 创建新的点云
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"Filtered point cloud has {len(filtered_cloud.points)} points")
    return filtered_cloud

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Point Cloud Processing")
    parser.add_argument("input_file", type=str, help="Input point cloud file path")
    parser.add_argument("output_file", type=str, help="Output point cloud file path")
    parser.add_argument("--density_radius", type=float, default=0.06, help="Search radius for density filtering")
    parser.add_argument("--min_density", type=int, default=10, help="Minimum local density for density filtering")
    parser.add_argument("--filter_type", type=str, default="statistical", choices=["statistical", "radius"], help="Filter type")
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Number of neighbors for statistical filter")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Standard deviation ratio for statistical filter")
    parser.add_argument("--radius_nb_points", type=int, default=16, help="Number of points for radius filter")
    parser.add_argument("--radius", type=float, default=0.1, help="Radius for radius filter")
    parser.add_argument("--normal_radius", type=float, default=0.1, help="Radius for normal estimation")
    parser.add_argument("--normal_max_nn", type=int, default=30, help="Maximum number of neighbors for normal estimation")
    parser.add_argument("--curvature_threshold", type=float, default=0.5, help="Curvature threshold for outlier removal")
    parser.add_argument("--cluster_eps", type=float, default=0.2, help="Epsilon for DBSCAN clustering")
    parser.add_argument("--cluster_min_points", type=int, default=20, help="Minimum points for DBSCAN clustering")
    parser.add_argument("--visualize", action="store_true", help="Visualize the point cloud at each step")
    
    args = parser.parse_args()
    
    # 加载点云
    point_cloud = load_point_cloud(args.input_file)
    
    if args.visualize:
        o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")
    
    # 1. 密度过滤 - 去除离散点，保留稠密区域（不进行下采样）
    down_sampled = density_filter(point_cloud, radius=args.density_radius, min_density=args.min_density)
    
    if args.visualize:
        o3d.visualization.draw_geometries([down_sampled], window_name="Down Sampled Point Cloud")
    
    # 2. 滤波
    if args.filter_type == "statistical":
        filtered = statistical_outlier_removal(down_sampled, args.nb_neighbors, args.std_ratio)
    else:
        filtered = radius_outlier_removal(down_sampled, args.radius_nb_points, args.radius)
    
    if args.visualize:
        o3d.visualization.draw_geometries([filtered], window_name="Filtered Point Cloud")
    
    # 3. 法线估计
    with_normals = estimate_normals(filtered, o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=args.normal_max_nn))
    
    if args.visualize:
        o3d.visualization.draw_geometries([with_normals], window_name="Point Cloud with Normals", point_show_normal=True)
    
    # 4. 基于法线/曲率的异常点剔除
    normal_filtered = remove_outliers_based_on_normals(with_normals, args.curvature_threshold)
    
    if args.visualize:
        o3d.visualization.draw_geometries([normal_filtered], window_name="Normal Filtered Point Cloud")
    
    # 5. 聚类分离小碎片
    final_cloud = cluster_and_remove_small_clusters(normal_filtered, args.cluster_eps, args.cluster_min_points)
    
    if args.visualize:
        o3d.visualization.draw_geometries([final_cloud], window_name="Final Point Cloud")
    
    # 保存处理后的点云
    print(f"Saving processed point cloud to {args.output_file}...")
    o3d.io.write_point_cloud(args.output_file, final_cloud)
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()