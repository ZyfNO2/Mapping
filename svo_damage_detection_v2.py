# -*- coding: utf-8 -*-
"""
SVO损伤检测与点云标记一体化脚本 V2

改进功能：
1. 对marked和original点云进行后处理（密度过滤、统计滤波、法线估计、曲率过滤、聚类）
2. damage_only点云不进行后处理
3. 可视化时点云水平放置（通过旋转矩阵调整视角）
4. 生成更多输出文件：
   - {svo_name}_original.ply: 原始重建点云（无标记，未处理）
   - {svo_name}_original_processed.ply: 原始点云（处理后）
   - {svo_name}_marked.ply: 带红色标记的完整点云（未处理）
   - {svo_name}_marked_processed.ply: 带标记点云（处理后）
   - {svo_name}_damage_only.ply: 仅包含损伤区域的点云（不处理）

使用方法：
    python svo_damage_detection_v2.py --svo "path/to/your.svo2"
    python svo_damage_detection_v2.py --svo "path/to/your.svo2" --model "path/to/best.pt"
    python svo_damage_detection_v2.py --svo "path/to/your.svo2" --visualize  # 可视化结果
"""

import sys
import time
import argparse
import pyzed.sl as sl
import os
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from collections import defaultdict


# 默认SVO文件路径
DEFAULT_SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-17-12.svo2"

# 默认模型路径
DEFAULT_MODEL_PATH = r"G:\Zed\spatial mapping\python\seg.pt"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "data/damage_detection"

# 检测置信度阈值
CONF_THRESHOLD = 0.15

# 标记颜色（红色）
MARK_COLOR = [1.0, 0.0, 0.0]


class DamageDetector:
    """损伤检测器类"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.15):
        """
        初始化损伤检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        print(f"[Info] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("[Info] Model loaded successfully!")
        
    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """
        检测图像中的损伤
        
        Args:
            image: BGR格式的图像
            
        Returns:
            masks: 检测到的mask列表
            confidences: 置信度列表
            boxes: 边界框列表
        """
        # 转换到RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 运行检测
        results = self.model(image_rgb, verbose=False, conf=self.conf_threshold)
        
        masks = []
        confidences = []
        boxes = []
        
        if results[0].masks is not None and len(results[0].masks) > 0:
            # 获取mask数据
            result_masks = results[0].masks.data.cpu().numpy()
            result_boxes = results[0].boxes.xyxy.cpu().numpy()
            result_confs = results[0].boxes.conf.cpu().numpy()
            
            # 过滤低置信度
            for mask, box, conf in zip(result_masks, result_boxes, result_confs):
                if conf >= self.conf_threshold:
                    # 调整mask尺寸到图像尺寸
                    mask_resized = cv2.resize(
                        mask, 
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    masks.append(mask_resized)
                    confidences.append(float(conf))
                    boxes.append(box)
        
        return masks, confidences, boxes


def extract_damage_points_from_frame(
    zed: sl.Camera,
    mask: np.ndarray,
    pose: sl.Pose,
    sample_step: int = 5
) -> List[np.ndarray]:
    """
    从单帧中提取损伤区域的3D点云（世界坐标系）
    """
    # 获取点云数据
    point_cloud_frame = sl.Mat()
    zed.retrieve_measure(point_cloud_frame, sl.MEASURE.XYZRGBA)
    
    # 获取mask区域的坐标
    mask_coords = np.where(mask > 0.5)
    mask_y_coords = mask_coords[0]
    mask_x_coords = mask_coords[1]
    
    # 提取位姿矩阵
    pose_data = pose.pose_data()
    rotation = np.array([
        [pose_data[0, 0], pose_data[0, 1], pose_data[0, 2]],
        [pose_data[1, 0], pose_data[1, 1], pose_data[1, 2]],
        [pose_data[2, 0], pose_data[2, 1], pose_data[2, 2]]
    ])
    translation = np.array([pose_data[0, 3], pose_data[1, 3], pose_data[2, 3]])
    
    damage_points = []
    
    # 采样提取点云
    for i in range(0, len(mask_y_coords), sample_step):
        y = mask_y_coords[i]
        x = mask_x_coords[i]
        
        # 获取该像素的3D坐标
        point_3d = point_cloud_frame.get_value(x, y)[1]
        
        # 检查有效点
        if (not np.isnan(point_3d[0]) and 
            not np.isnan(point_3d[1]) and 
            not np.isnan(point_3d[2]) and
            abs(point_3d[0]) < 10 and 
            abs(point_3d[1]) < 10 and 
            abs(point_3d[2]) < 10):
            
            # 转换到世界坐标系
            point_camera = np.array([point_3d[0], point_3d[1], point_3d[2]])
            point_world = rotation @ point_camera + translation
            
            damage_points.append(point_world)
    
    point_cloud_frame.free(sl.MEM.CPU)
    
    return damage_points


# ==================== 点云处理函数 ====================

def density_filter(point_cloud, radius=0.06, min_density=10):
    """密度过滤 - 去除离散点，保留稠密区域"""
    print(f"[Processing] Density filtering with radius: {radius}, min density: {min_density}...")
    
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    dense_points = []
    dense_colors = []
    
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    
    for i, point in enumerate(points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if len(idx) >= min_density:
            dense_points.append(point)
            if colors is not None:
                dense_colors.append(colors[i])
    
    dense_cloud = o3d.geometry.PointCloud()
    dense_cloud.points = o3d.utility.Vector3dVector(np.array(dense_points))
    if dense_colors:
        dense_cloud.colors = o3d.utility.Vector3dVector(np.array(dense_colors))
    
    print(f"[Processing] Density filtered: {len(dense_cloud.points)} points")
    return dense_cloud


def statistical_outlier_removal(point_cloud, nb_neighbors=20, std_ratio=2.0):
    """统计滤波"""
    print(f"[Processing] Statistical outlier removal...")
    filtered, indices = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[Processing] After statistical filter: {len(filtered.points)} points")
    return filtered


def estimate_normals(point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """法线估计"""
    print(f"[Processing] Estimating normals...")
    point_cloud.estimate_normals(search_param=search_param)
    return point_cloud


def remove_outliers_based_on_normals(point_cloud, threshold=0.5):
    """基于法线/曲率的异常点剔除"""
    print(f"[Processing] Removing outliers based on normals/curvature...")
    
    point_cloud.estimate_normals()
    curvatures = []
    
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    for i in range(len(point_cloud.points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(point_cloud.points[i], 10)
        if len(idx) > 1:
            neighbors = np.asarray(point_cloud.points)[idx[1:], :]
            center = np.asarray(point_cloud.points)[idx[0], :]
            cov = np.cov(neighbors - center, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(cov)
            curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-10)
            curvatures.append(curvature)
        else:
            curvatures.append(1.0)
    
    curvatures = np.array(curvatures)
    mask = curvatures < threshold
    filtered_points = np.asarray(point_cloud.points)[mask]
    filtered_colors = np.asarray(point_cloud.colors)[mask] if point_cloud.has_colors() else None
    
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"[Processing] After normal filter: {len(filtered_cloud.points)} points")
    return filtered_cloud


def cluster_and_remove_small_clusters(point_cloud, eps=0.1, min_points=50):
    """聚类分离小碎片"""
    print(f"[Processing] Clustering with eps={eps}, min_points={min_points}...")
    
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    max_label = labels.max()
    
    if max_label < 0:
        print(f"[Processing] No clusters found, returning original")
        return point_cloud
    
    cluster_sizes = np.bincount(labels[labels >= 0])
    large_clusters = np.where(cluster_sizes >= min_points)[0]
    
    if len(large_clusters) == 0:
        print(f"[Processing] No large clusters found, returning original")
        return point_cloud
    
    mask = np.isin(labels, large_clusters)
    filtered_points = np.asarray(point_cloud.points)[mask]
    filtered_colors = np.asarray(point_cloud.colors)[mask] if point_cloud.has_colors() else None
    
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"[Processing] After clustering: {len(filtered_cloud.points)} points ({len(large_clusters)} clusters)")
    return filtered_cloud


def process_point_cloud(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    处理点云：密度过滤 -> 统计滤波 -> 法线估计 -> 曲率过滤 -> 聚类
    
    Args:
        point_cloud: 输入点云
        
    Returns:
        处理后的点云
    """
    print("\n" + "="*50)
    print("Starting point cloud processing pipeline")
    print("="*50)
    
    original_count = len(point_cloud.points)
    print(f"[Input] Point cloud has {original_count} points")
    
    # 1. 密度过滤
    cloud = density_filter(point_cloud, radius=0.06, min_density=10)
    
    # 2. 统计滤波
    cloud = statistical_outlier_removal(cloud, nb_neighbors=20, std_ratio=2.0)
    
    # 3. 法线估计
    cloud = estimate_normals(cloud)
    
    # 4. 基于法线/曲率的异常点剔除
    cloud = remove_outliers_based_on_normals(cloud, threshold=0.5)
    
    # 5. 聚类分离小碎片
    cloud = cluster_and_remove_small_clusters(cloud, eps=0.1, min_points=50)
    
    final_count = len(cloud.points)
    reduction = (1 - final_count / original_count) * 100 if original_count > 0 else 0
    
    print("="*50)
    print(f"Processing complete: {original_count} -> {final_count} points ({reduction:.1f}% reduction)")
    print("="*50 + "\n")
    
    return cloud


# ==================== 可视化函数 ====================

def align_point_cloud_to_horizontal(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    将点云对齐到水平面（通过PCA计算主平面并旋转）
    
    Args:
        pcd: 输入点云
        
    Returns:
        旋转后的点云
    """
    points = np.asarray(pcd.points)
    
    # 计算质心
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # PCA计算主成分
    cov = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 按特征值排序（从小到大）
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 最小特征值对应的特征向量是法线方向（垂直于主平面）
    # 我们希望这个方向与Z轴对齐
    normal = eigenvectors[:, 0]
    
    # 计算旋转矩阵，使法线对齐到Z轴
    z_axis = np.array([0, 0, 1])
    
    # 如果法线已经近似Z轴，不需要旋转
    if np.abs(np.dot(normal, z_axis)) > 0.99:
        return pcd
    
    # 计算旋转轴和角度
    rotation_axis = np.cross(normal, z_axis)
    rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-10)
    
    cos_angle = np.dot(normal, z_axis)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # 构建旋转矩阵（Rodrigues公式）
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # 应用旋转
    points_rotated = (R @ points_centered.T).T + centroid
    
    # 创建新的点云
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_rotated)
    if pcd.has_colors():
        pcd_aligned.colors = pcd.colors
    if pcd.has_normals():
        normals_rotated = (R @ np.asarray(pcd.normals).T).T
        pcd_aligned.normals = o3d.utility.Vector3dVector(normals_rotated)
    
    return pcd_aligned


def visualize_point_clouds(
    original_pcd: o3d.geometry.PointCloud,
    marked_pcd: o3d.geometry.PointCloud,
    damage_only_pcd: o3d.geometry.PointCloud,
    original_processed: o3d.geometry.PointCloud = None,
    marked_processed: o3d.geometry.PointCloud = None,
    align_horizontal: bool = True
):
    """
    可视化点云对比
    
    Args:
        original_pcd: 原始点云
        marked_pcd: 标记点云
        damage_only_pcd: 仅损伤点云
        original_processed: 处理后的原始点云
        marked_processed: 处理后的标记点云
        align_horizontal: 是否将点云水平放置
    """
    print("\n[Visualization] Preparing point clouds for visualization...")
    
    # 如果需要水平对齐
    if align_horizontal:
        print("[Visualization] Aligning point clouds to horizontal...")
        original_pcd = align_point_cloud_to_horizontal(original_pcd)
        marked_pcd = align_point_cloud_to_horizontal(marked_pcd)
        damage_only_pcd = align_point_cloud_to_horizontal(damage_only_pcd)
        if original_processed is not None:
            original_processed = align_point_cloud_to_horizontal(original_processed)
        if marked_processed is not None:
            marked_processed = align_point_cloud_to_horizontal(marked_processed)
    
    # 创建可视化窗口
    vis_list = []
    
    # 1. 原始点云
    original_pcd_copy = copy_point_cloud(original_pcd)
    vis_list.append(("Original Point Cloud", original_pcd_copy))
    
    # 2. 标记点云
    marked_pcd_copy = copy_point_cloud(marked_pcd)
    vis_list.append(("Marked Point Cloud", marked_pcd_copy))
    
    # 3. 仅损伤点云
    damage_only_copy = copy_point_cloud(damage_only_pcd)
    vis_list.append(("Damage Only Point Cloud", damage_only_copy))
    
    # 4. 处理后的原始点云
    if original_processed is not None:
        orig_proc_copy = copy_point_cloud(original_processed)
        vis_list.append(("Original Processed", orig_proc_copy))
    
    # 5. 处理后的标记点云
    if marked_processed is not None:
        marked_proc_copy = copy_point_cloud(marked_processed)
        vis_list.append(("Marked Processed", marked_proc_copy))
    
    # 显示所有点云
    for title, pcd in vis_list:
        print(f"[Visualization] Showing: {title} ({len(pcd.points)} points)")
        o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)


def copy_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """复制点云"""
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        pcd_copy.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    return pcd_copy


# ==================== 主处理函数 ====================

def process_svo_with_damage_detection_v2(
    svo_path: str,
    model_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    conf_threshold: float = CONF_THRESHOLD,
    save_detection_images: bool = True,
    visualize: bool = False,
    skip_processing: bool = False
):
    """
    处理SVO文件，进行损伤检测、点云标记和后处理
    
    Args:
        svo_path: SVO文件路径
        model_path: YOLO模型路径
        output_dir: 输出目录
        conf_threshold: 检测置信度阈值
        save_detection_images: 是否保存检测图像
        visualize: 是否可视化结果
        skip_processing: 是否跳过点云处理（用于快速测试）
    """
    print("="*70)
    print("SVO Damage Detection & Point Cloud Marking V2")
    print("="*70)
    
    # 获取SVO文件名
    svo_name = Path(svo_path).stem
    output_dir = os.path.join(output_dir, svo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if save_detection_images:
        detection_images_dir = os.path.join(output_dir, "detection_images")
        os.makedirs(detection_images_dir, exist_ok=True)
    
    print(f"[Info] Output directory: {output_dir}")
    print(f"[Info] Using SVO: {svo_path}")
    print(f"[Info] Using model: {model_path}")
    print(f"[Info] Skip processing: {skip_processing}")
    
    # 初始化损伤检测器
    detector = DamageDetector(model_path, conf_threshold)
    
    # 初始化ZED
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    
    init.set_from_svo_file(svo_path)
    
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Camera Open: {repr(status)}. Exit program.")
        return False
    
    camera_info = zed.get_camera_information()
    image_size = camera_info.camera_configuration.resolution
    print(f"[Info] Camera resolution: {image_size.width}x{image_size.height}")
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Enable Positional Tracking: {repr(returned_state)}. Exit program.")
        zed.close()
        return False
    
    # 配置空间映射
    print("[Info] Using HIGH quality point cloud settings with SHORT range")
    spatial_mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.HIGH,
        mapping_range=sl.MAPPING_RANGE.SHORT,
        max_memory_usage=8192,
        save_texture=True,
        use_chunk_only=False,
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    )
    
    spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
    spatial_mapping_parameters.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.SHORT)
    spatial_mapping_parameters.save_texture = True
    spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    
    zed.enable_spatial_mapping(spatial_mapping_parameters)
    
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    pose = sl.Pose()
    image = sl.Mat()
    
    # 存储所有损伤点（世界坐标系）
    all_damage_points = []
    damage_frame_count = 0
    total_detections = 0
    
    print("\n[Info] Processing SVO file with damage detection...")
    frame_count = 0
    
    while True:
        grab_result = zed.grab(runtime_parameters)
        
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed.")
            break
        
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            zed.get_position(pose)
            
            # 获取图像
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            
            # 转换为BGR格式
            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame
            
            # 运行损伤检测
            masks, confidences, boxes = detector.detect(frame_bgr)
            
            if len(masks) > 0:
                damage_frame_count += 1
                total_detections += len(masks)
                
                print(f"[Frame {frame_count}] Detected {len(masks)} damage regions")
                
                # 提取每个mask区域的点云
                for i, (mask, conf, box) in enumerate(zip(masks, confidences, boxes)):
                    damage_points = extract_damage_points_from_frame(zed, mask, pose)
                    
                    if len(damage_points) > 0:
                        all_damage_points.extend(damage_points)
                        print(f"  - Damage {i+1}: {len(damage_points)} points (conf: {conf:.3f})")
                
                # 保存检测图像
                if save_detection_images:
                    result_image = frame_bgr.copy()
                    for mask, conf in zip(masks, confidences):
                        mask_bool = mask > 0.5
                        colored_mask = np.zeros_like(result_image)
                        colored_mask[mask_bool] = [0, 0, 255]
                        result_image[mask_bool] = cv2.addWeighted(
                            result_image[mask_bool], 0.5,
                            colored_mask[mask_bool], 0.5, 0
                        )
                    
                    image_path = os.path.join(detection_images_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(image_path, result_image)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    print(f"\n[Info] Detection summary:")
    print(f"  - Total frames: {frame_count}")
    print(f"  - Frames with damage: {damage_frame_count}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Total damage points: {len(all_damage_points)}")
    
    # 提取完整点云
    print("\n[Info] Extracting full point cloud...")
    point_cloud = sl.FusedPointCloud()
    extract_status = zed.extract_whole_spatial_map(point_cloud)
    
    if extract_status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to extract point cloud: {extract_status}")
        zed.close()
        return False
    
    print("[Info] Successfully extracted point cloud")
    
    # 保存原始点云
    original_ply_path = os.path.join(output_dir, f"{svo_name}_original.ply")
    point_cloud.save(original_ply_path, sl.MESH_FILE_FORMAT.PLY)
    print(f"[Info] Saved original point cloud: {original_ply_path}")
    
    # 使用Open3D处理点云并添加标记
    try:
        pcd = o3d.io.read_point_cloud(original_ply_path)
        original_points = np.asarray(pcd.points)
        
        print(f"[Info] Full point cloud has {len(original_points)} points")
        
        # 创建标记点云
        marked_pcd = copy_point_cloud(pcd)
        
        if len(all_damage_points) > 0:
            damage_points_np = np.array(all_damage_points)
            
            # 创建颜色数组
            if pcd.has_colors():
                original_colors = np.asarray(pcd.colors)
            else:
                original_colors = np.ones((len(original_points), 3)) * 0.7
            
            # 使用KDTree匹配损伤点
            print("[Info] Matching damage points with full point cloud...")
            from scipy.spatial import cKDTree
            damage_tree = cKDTree(damage_points_np)
            
            threshold = 0.05  # 5cm阈值
            marked_count = 0
            
            for i, pt in enumerate(original_points):
                dist, idx = damage_tree.query(pt, distance_upper_bound=threshold)
                if dist != np.inf:
                    original_colors[i] = MARK_COLOR
                    marked_count += 1
            
            print(f"[Info] Marked {marked_count} points as damage regions")
            marked_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        else:
            print("[Warning] No damage points detected")
            if not pcd.has_colors():
                marked_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(original_points), 3)) * 0.7)
        
        # 保存标记点云（未处理）
        marked_ply_path = os.path.join(output_dir, f"{svo_name}_marked.ply")
        o3d.io.write_point_cloud(marked_ply_path, marked_pcd)
        print(f"[Info] Saved marked point cloud: {marked_ply_path}")
        
        # 保存仅损伤区域的点云（不进行处理）
        if len(all_damage_points) > 0:
            damage_only_pcd = o3d.geometry.PointCloud()
            damage_only_pcd.points = o3d.utility.Vector3dVector(damage_points_np)
            damage_colors = np.ones((len(damage_points_np), 3)) * MARK_COLOR
            damage_only_pcd.colors = o3d.utility.Vector3dVector(damage_colors)
            
            damage_only_path = os.path.join(output_dir, f"{svo_name}_damage_only.ply")
            o3d.io.write_point_cloud(damage_only_path, damage_only_pcd)
            print(f"[Info] Saved damage-only point cloud: {damage_only_path}")
        
        # ==================== 点云处理 ====================
        if not skip_processing:
            print("\n" + "="*70)
            print("Starting point cloud post-processing")
            print("="*70)
            
            # 处理原始点云
            print("\n[Processing] Processing original point cloud...")
            original_processed = process_point_cloud(pcd)
            original_processed_path = os.path.join(output_dir, f"{svo_name}_original_processed.ply")
            o3d.io.write_point_cloud(original_processed_path, original_processed)
            print(f"[Info] Saved processed original: {original_processed_path}")
            
            # 处理标记点云
            print("\n[Processing] Processing marked point cloud...")
            marked_processed = process_point_cloud(marked_pcd)
            marked_processed_path = os.path.join(output_dir, f"{svo_name}_marked_processed.ply")
            o3d.io.write_point_cloud(marked_processed_path, marked_processed)
            print(f"[Info] Saved processed marked: {marked_processed_path}")
            
            print("\n" + "="*70)
            print("Point cloud processing completed")
            print("="*70)
        else:
            print("\n[Info] Skipping point cloud processing (skip_processing=True)")
            original_processed = None
            marked_processed = None
        
        # ==================== 可视化 ====================
        if visualize:
            print("\n[Info] Starting visualization...")
            
            # 准备点云
            if len(all_damage_points) > 0:
                damage_only_pcd_vis = damage_only_pcd
            else:
                # 创建空点云
                damage_only_pcd_vis = o3d.geometry.PointCloud()
            
            visualize_point_clouds(
                original_pcd=pcd,
                marked_pcd=marked_pcd,
                damage_only_pcd=damage_only_pcd_vis,
                original_processed=original_processed,
                marked_processed=marked_processed,
                align_horizontal=True
            )
            
    except Exception as e:
        print(f"[Error] Failed to process point cloud: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    print("\n" + "="*70)
    print("Processing completed!")
    print(f"Output files saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {svo_name}_original.ply (原始点云)")
    if not skip_processing:
        print(f"  - {svo_name}_original_processed.ply (原始点云-处理后)")
    print(f"  - {svo_name}_marked.ply (标记点云)")
    if not skip_processing:
        print(f"  - {svo_name}_marked_processed.ply (标记点云-处理后)")
    print(f"  - {svo_name}_damage_only.ply (仅损伤区域)")
    print("="*70)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SVO Damage Detection & Point Cloud Marking V2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python svo_damage_detection_v2.py --svo "path\\to\\your.svo2"
  python svo_damage_detection_v2.py --svo "path\\to\\your.svo2" --model "path\\to\\best.pt"
  python svo_damage_detection_v2.py --svo "path\\to\\your.svo2" --conf 0.2
  python svo_damage_detection_v2.py --svo "path\\to\\your.svo2" --visualize
  python svo_damage_detection_v2.py --svo "path\\to\\your.svo2" --skip-processing  # 跳过点云处理
        """
    )
    
    parser.add_argument(
        '--svo',
        type=str,
        default=DEFAULT_SVO_PATH,
        help=f'Path to SVO file (default: {DEFAULT_SVO_PATH})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to YOLO model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=CONF_THRESHOLD,
        help=f'Confidence threshold (default: {CONF_THRESHOLD})'
    )
    
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Do not save detection images'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize point clouds after processing'
    )
    
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip point cloud post-processing (for faster testing)'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.svo):
        print(f"[Error] SVO file not found: {args.svo}")
        return
    
    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        return
    
    # 运行处理
    process_svo_with_damage_detection_v2(
        svo_path=args.svo,
        model_path=args.model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        save_detection_images=not args.no_images,
        visualize=args.visualize,
        skip_processing=args.skip_processing
    )


if __name__ == "__main__":
    main()
