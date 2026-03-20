# -*- coding: utf-8 -*-
"""
在点云重建流程中标记特定帧mask覆盖的区域 - V2版本

功能：
1. 读取指定帧的mask JSON文件
2. 在SVO处理过程中，对该帧的mask区域点云进行红色标记
3. 生成三种点云文件：
   - frame_XXXXXX_original.ply: 原始重建点云（无标记）
   - frame_XXXXXX_marked.ply: 带红色标记的完整点云
   - frame_XXXXXX_mask_only.ply: 仅包含mask区域的点云

使用方法：
    python mask_point_cloud_marker_v2.py --frame 33 --svo "path/to/your.svo2"
    python mask_point_cloud_marker_v2.py --frame 33  # 使用默认SVO文件
"""

import sys
import time
import argparse
import pyzed.sl as sl
import os
import open3d as o3d
import numpy as np
import json
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


# 默认SVO文件路径
DEFAULT_SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-17-12.svo2"

# 默认分割输出目录
DEFAULT_SEGMENTATION_DIR = r"G:\Zed\spatial mapping\output_segmentation_filtered"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "data/mask_marked"


def extract_mask_from_segmented_image(segmented_image_path: str) -> Optional[np.ndarray]:
    """
    从分割后的图像中提取mask（基于红色标记区域）
    """
    try:
        segmented = cv2.imread(segmented_image_path)
        if segmented is None:
            print(f"[Warning] Failed to load segmented image: {segmented_image_path}")
            return None
        
        hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        
        # 红色的HSV范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        return red_mask
        
    except Exception as e:
        print(f"[Error] Failed to extract mask: {e}")
        return None


def create_mask_from_bbox(mask_data: dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    从bbox信息创建mask
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for mask_info in mask_data.get('masks', []):
        bbox = mask_info['bbox']
        x1, y1 = max(0, bbox['x1']), max(0, bbox['y1'])
        x2, y2 = min(width, bbox['x2']), min(height, bbox['y2'])
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    
    return mask


def process_svo_with_mask_marking(
    svo_path: str,
    frame_id: int,
    mask_json_path: str,
    segmented_image_path: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    mark_color: List[float] = [1.0, 0.0, 0.0]  # 红色
):
    """
    处理SVO文件，在指定帧的mask区域标记点云颜色
    """
    print("="*70)
    print(f"Mask Point Cloud Marker V2 - Frame {frame_id}")
    print("="*70)
    
    # 加载mask数据
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    
    actual_frame_id = mask_data.get('frame_id', frame_id * 4)
    print(f"[Info] Processing actual frame ID: {actual_frame_id}")
    
    # 初始化ZED
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    
    init.set_from_svo_file(svo_path)
    print(f"[Info] Using SVO File: {svo_path}")

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
    
    # 加载mask
    mask_full = None
    if segmented_image_path and os.path.exists(segmented_image_path):
        print(f"[Info] Loading mask from segmented image: {segmented_image_path}")
        mask_full = extract_mask_from_segmented_image(segmented_image_path)
    
    if mask_full is None:
        print("[Info] Using bbox-based mask approximation")
        mask_full = create_mask_from_bbox(mask_data, (image_size.height, image_size.width))
    
    # 调整mask尺寸
    if mask_full.shape[0] != image_size.height or mask_full.shape[1] != image_size.width:
        print(f"[Info] Resizing mask from {mask_full.shape} to ({image_size.height}, {image_size.width})")
        mask_full = cv2.resize(mask_full, (image_size.width, image_size.height), interpolation=cv2.INTER_NEAREST)
    
    mask_coords = np.where(mask_full > 0)
    mask_y_coords = mask_coords[0]
    mask_x_coords = mask_coords[1]
    
    print(f"[Info] Mask covers {len(mask_y_coords)} pixels")
    
    # 存储mask区域的3D点（在世界坐标系中）
    mask_world_points = []
    mask_colors = []
    
    sample_step = 5
    current_frame = 0
    print("Processing SVO file...")
    
    while True:
        grab_result = zed.grab(runtime_parameters)
        
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed.")
            break
            
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            zed.get_position(pose)
            
            if current_frame == actual_frame_id:
                print(f"[Info] Processing target frame {current_frame}")
                
                # 获取点云数据
                point_cloud_frame = sl.Mat()
                zed.retrieve_measure(point_cloud_frame, sl.MEASURE.XYZRGBA)
                
                # 在mask区域采样点云
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
                        
                        # 将点从相机坐标系转换到世界坐标系
                        # 使用位姿矩阵进行变换
                        pose_data = pose.pose_data()
                        
                        # 从Matrix4f中提取旋转矩阵和平移向量
                        rotation = np.array([
                            [pose_data[0, 0], pose_data[0, 1], pose_data[0, 2]],
                            [pose_data[1, 0], pose_data[1, 1], pose_data[1, 2]],
                            [pose_data[2, 0], pose_data[2, 1], pose_data[2, 2]]
                        ])
                        translation = np.array([pose_data[0, 3], pose_data[1, 3], pose_data[2, 3]])
                        
                        # 变换点坐标
                        point_camera = np.array([point_3d[0], point_3d[1], point_3d[2]])
                        point_world = rotation @ point_camera + translation
                        
                        mask_world_points.append(point_world)
                        mask_colors.append(mark_color)
                
                print(f"[Info] Collected {len(mask_world_points)} mask points in world coordinates")
                
                point_cloud_frame.free(sl.MEM.CPU)
            
            current_frame += 1
            
            if current_frame % 100 == 0:
                print(f"Processed {current_frame} frames...")

    # 提取完整点云
    print("\nExtracting full point cloud...")
    point_cloud = sl.FusedPointCloud()
    extract_status = zed.extract_whole_spatial_map(point_cloud)
    
    if extract_status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to extract point cloud: {extract_status}")
        zed.close()
        return False
    
    print("Successfully extracted point cloud")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始点云
    original_ply_path = os.path.join(output_dir, f"frame_{frame_id:06d}_original.ply")
    point_cloud.save(original_ply_path, sl.MESH_FILE_FORMAT.PLY)
    print(f"[Info] Saved original point cloud: {original_ply_path}")
    
    # 使用Open3D处理点云
    try:
        # 读取原始点云
        pcd = o3d.io.read_point_cloud(original_ply_path)
        original_points = np.asarray(pcd.points)
        
        if len(mask_world_points) > 0:
            mask_points_np = np.array(mask_world_points)
            mask_colors_np = np.array(mask_colors)
            
            print(f"[Info] Original point cloud has {len(original_points)} points")
            print(f"[Info] Mask region has {len(mask_points_np)} points")
            
            # 创建原始点云的颜色数组
            if pcd.has_colors():
                original_colors = np.asarray(pcd.colors)
            else:
                # 如果没有颜色，使用默认灰色
                original_colors = np.ones((len(original_points), 3)) * 0.7
            
            # 使用KDTree进行近邻搜索来标记点云
            print("[Info] Building KDTree for mask points...")
            from scipy.spatial import cKDTree
            mask_tree = cKDTree(mask_points_np)
            
            # 对于每个原始点云点，查找最近的mask点
            print("[Info] Matching points...")
            threshold = 0.05  # 5cm阈值
            marked_count = 0
            
            for i, pt in enumerate(original_points):
                dist, idx = mask_tree.query(pt, distance_upper_bound=threshold)
                if dist != np.inf:  # 找到了邻近点
                    original_colors[i] = mark_color
                    marked_count += 1
            
            print(f"[Info] Marked {marked_count} points in the full point cloud")
            
            # 保存带标记的点云
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            marked_ply_path = os.path.join(output_dir, f"frame_{frame_id:06d}_marked.ply")
            o3d.io.write_point_cloud(marked_ply_path, pcd)
            print(f"[Info] Saved marked point cloud: {marked_ply_path}")
            
            # 保存仅包含mask区域的点云
            mask_only_pcd = o3d.geometry.PointCloud()
            mask_only_pcd.points = o3d.utility.Vector3dVector(mask_points_np)
            mask_only_pcd.colors = o3d.utility.Vector3dVector(mask_colors_np)
            
            mask_only_path = os.path.join(output_dir, f"frame_{frame_id:06d}_mask_only.ply")
            o3d.io.write_point_cloud(mask_only_path, mask_only_pcd)
            print(f"[Info] Saved mask-only point cloud: {mask_only_path}")
        else:
            print("[Warning] No mask points collected")
            
    except Exception as e:
        print(f"[Error] Failed to process point cloud: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    print("\nProcessing completed!")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Mark mask regions in point cloud with red color',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_point_cloud_marker_v2.py --frame 33
  python mask_point_cloud_marker_v2.py --frame 33 --svo "path\\to\\your.svo2"
        """
    )
    
    parser.add_argument('--frame', type=int, required=True, help='Frame ID to process')
    parser.add_argument('--svo', type=str, default=DEFAULT_SVO_PATH, help='Path to SVO file')
    parser.add_argument('--segmentation-dir', type=str, default=DEFAULT_SEGMENTATION_DIR, help='Segmentation output directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    mask_json_path = os.path.join(args.segmentation_dir, 'masks', f'frame_{args.frame:06d}_masks.json')
    segmented_image_path = os.path.join(args.segmentation_dir, 'segmented', f'frame_{args.frame:06d}_segmented.jpg')
    
    if not os.path.exists(mask_json_path):
        print(f"[Error] Mask JSON file not found: {mask_json_path}")
        return
    
    print(f"[Info] Found mask file: {mask_json_path}")
    
    if os.path.exists(segmented_image_path):
        print(f"[Info] Found segmented image: {segmented_image_path}")
    else:
        print(f"[Warning] Segmented image not found, will use bbox approximation")
        segmented_image_path = None
    
    process_svo_with_mask_marking(
        svo_path=args.svo,
        frame_id=args.frame,
        mask_json_path=mask_json_path,
        segmented_image_path=segmented_image_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
