# -*- coding: utf-8 -*-
"""
在点云重建流程中标记特定帧mask覆盖的区域

功能：
1. 读取指定帧的mask JSON文件
2. 在SVO处理过程中，对该帧的mask区域点云进行红色标记
3. 生成带颜色标记的点云文件

使用方法：
    python mask_point_cloud_marker.py --frame 33 --svo "path/to/your.svo2"
    python mask_point_cloud_marker.py --frame 33  # 使用默认SVO文件
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


def load_mask_from_json(mask_json_path: str) -> Optional[np.ndarray]:
    """
    从JSON文件加载mask数据
    
    Args:
        mask_json_path: mask JSON文件路径
        
    Returns:
        np.ndarray: 二值mask图像，如果加载失败返回None
    """
    try:
        with open(mask_json_path, 'r') as f:
            mask_data = json.load(f)
        
        if not mask_data.get('masks') or len(mask_data['masks']) == 0:
            print(f"[Warning] No masks found in {mask_json_path}")
            return None
        
        # 获取第一个mask的信息
        mask_info = mask_data['masks'][0]
        shape = mask_info['shape']  # [height, width]
        bbox = mask_info['bbox']    # {x1, y1, x2, y2}
        
        # 创建全零mask
        mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        
        # 在bbox区域内填充mask（简化处理，实际应该使用轮廓或RLE解码）
        # 这里我们使用bbox作为近似mask区域
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        mask[y1:y2, x1:x2] = 255
        
        print(f"[Info] Loaded mask with shape {shape}, bbox: ({x1}, {y1}) - ({x2}, {y2})")
        return mask
        
    except Exception as e:
        print(f"[Error] Failed to load mask from {mask_json_path}: {e}")
        return None


def create_full_mask_from_bbox(mask_data: dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    从bbox信息创建完整尺寸的mask
    
    Args:
        mask_data: mask数据字典
        image_shape: (height, width)
        
    Returns:
        np.ndarray: 完整尺寸的二值mask
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


def extract_mask_from_segmented_image(segmented_image_path: str, red_threshold: int = 100) -> Optional[np.ndarray]:
    """
    从分割后的图像中提取mask（基于红色标记区域）
    
    Args:
        segmented_image_path: 分割图像路径
        red_threshold: 红色通道阈值
        
    Returns:
        np.ndarray: 二值mask图像，如果加载失败返回None
    """
    try:
        # 读取分割图像
        segmented = cv2.imread(segmented_image_path)
        if segmented is None:
            print(f"[Warning] Failed to load segmented image: {segmented_image_path}")
            return None
        
        # 转换到HSV色彩空间，更容易提取红色
        hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        
        # 红色的HSV范围（有两个范围，因为红色在HSV中跨越0度）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建红色mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        print(f"[Info] Extracted mask from segmented image, mask area: {np.sum(red_mask > 0)} pixels")
        return red_mask
        
    except Exception as e:
        print(f"[Error] Failed to extract mask from segmented image: {e}")
        return None


def process_svo_with_mask_marking(
    svo_path: str,
    frame_id: int,
    mask_json_path: str,
    segmented_image_path: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    mark_color: List[float] = [1.0, 0.0, 0.0]  # 红色 (RGB)
):
    """
    处理SVO文件，在指定帧的mask区域标记点云颜色
    
    Args:
        svo_path: SVO文件路径
        frame_id: 要处理的帧ID（在过滤后的序列中的ID）
        mask_json_path: mask JSON文件路径
        segmented_image_path: 分割图像路径（可选，如果提供会从图像中提取更精确的mask）
        output_dir: 输出目录
        mark_color: 标记颜色 [R, G, B]，默认红色
    """
    print("="*70)
    print(f"Mask Point Cloud Marker - Frame {frame_id}")
    print("="*70)
    
    # 加载mask数据
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    
    actual_frame_id = mask_data.get('frame_id', frame_id * 4)  # 获取实际帧ID
    print(f"[Info] Processing actual frame ID: {actual_frame_id}")
    
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    
    init.set_from_svo_file(svo_path)
    print(f"[Info] Using SVO File: {svo_path}")

    # 打开相机
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Camera Open: {repr(status)}. Exit program.")
        return False
    
    # 获取相机信息
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

    # 配置空间映射参数
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

    # 启用空间映射
    zed.enable_spatial_mapping(spatial_mapping_parameters)

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    mapping_activated = True
    image = sl.Mat()
    pose = sl.Pose()
    
    # 存储带标记的点云数据
    marked_points = []
    marked_colors = []
    
    # 尝试从segmented图像提取mask，如果失败则使用bbox
    mask_full = None
    if segmented_image_path and os.path.exists(segmented_image_path):
        print(f"[Info] Loading mask from segmented image: {segmented_image_path}")
        mask_full = extract_mask_from_segmented_image(segmented_image_path)
    
    if mask_full is None:
        print("[Info] Using bbox-based mask approximation")
        mask_full = create_full_mask_from_bbox(mask_data, (image_size.height, image_size.width))
    
    # 调整mask尺寸以匹配相机分辨率
    if mask_full.shape[0] != image_size.height or mask_full.shape[1] != image_size.width:
        print(f"[Info] Resizing mask from {mask_full.shape} to ({image_size.height}, {image_size.width})")
        mask_full = cv2.resize(mask_full, (image_size.width, image_size.height), interpolation=cv2.INTER_NEAREST)
    
    # 获取mask区域的坐标
    mask_coords = np.where(mask_full > 0)
    mask_y_coords = mask_coords[0]
    mask_x_coords = mask_coords[1]
    
    print(f"[Info] Mask covers {len(mask_y_coords)} pixels")
    
    # 创建mask点云集合用于快速查找（使用空间哈希）
    mask_points_set = set()
    
    # 用于采样（减少计算量）
    sample_step = 5  # 每5个像素采样一个点

    # 主循环
    current_frame = 0
    print("Processing SVO file...")
    
    while True:
        # 抓取图像
        grab_result = zed.grab(runtime_parameters)
        
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed.")
            break
            
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            # 更新位姿数据
            zed.get_position(pose)
            
            # 检查是否是目标帧
            if current_frame == actual_frame_id:
                print(f"[Info] Processing target frame {current_frame}")
                
                # 获取图像和点云数据
                image_zed = sl.Mat()
                point_cloud_frame = sl.Mat()
                
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud_frame, sl.MEASURE.XYZRGBA)
                
                # 获取图像数据
                image_ocv = image_zed.get_data()
                
                # 在mask区域采样点云
                for i in range(0, len(mask_y_coords), sample_step):
                    y = mask_y_coords[i]
                    x = mask_x_coords[i]
                    
                    # 获取该像素的3D坐标
                    point_3d = point_cloud_frame.get_value(x, y)[1]
                    
                    # 检查是否是有效点（非NaN且在一定范围内）
                    if (not np.isnan(point_3d[0]) and 
                        not np.isnan(point_3d[1]) and 
                        not np.isnan(point_3d[2]) and
                        abs(point_3d[0]) < 10 and 
                        abs(point_3d[1]) < 10 and 
                        abs(point_3d[2]) < 10):
                        
                        # 添加到列表
                        marked_points.append([point_3d[0], point_3d[1], point_3d[2]])
                        marked_colors.append(mark_color)
                        
                        # 添加到空间哈希集合（用于后续匹配）
                        # 使用量化坐标作为键（1cm精度）
                        key = (round(point_3d[0], 2), round(point_3d[1], 2), round(point_3d[2], 2))
                        mask_points_set.add(key)
                
                print(f"[Info] Collected {len(marked_points)} marked points from mask region")
                print(f"[Info] Mask points set size: {len(mask_points_set)}")
                
                # 释放资源
                image_zed.free(sl.MEM.CPU)
                point_cloud_frame.free(sl.MEM.CPU)
            
            current_frame += 1
            
            # 显示进度
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
    
    # 使用Open3D处理点云并添加颜色标记
    try:
        # 读取原始点云
        pcd = o3d.io.read_point_cloud(original_ply_path)
        
        if len(marked_points) > 0 and len(mask_points_set) > 0:
            # 将标记点转换为numpy数组
            marked_points_np = np.array(marked_points)
            marked_colors_np = np.array(marked_colors)
            
            # 获取原始点云数据
            original_points = np.asarray(pcd.points)
            original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(original_points), 3)) * 0.7
            
            # 使用空间哈希匹配点云
            print("[Info] Matching mask points with full point cloud using spatial hashing...")
            marked_count = 0
            
            for i, pt in enumerate(original_points):
                # 使用量化坐标作为键（1cm精度）
                key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
                if key in mask_points_set:
                    original_colors[i] = mark_color
                    marked_count += 1
            
            print(f"[Info] Marked {marked_count} points in the full point cloud")
            
            # 更新点云颜色
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            
            # 保存带标记的点云
            marked_ply_path = os.path.join(output_dir, f"frame_{frame_id:06d}_marked.ply")
            o3d.io.write_point_cloud(marked_ply_path, pcd)
            print(f"[Info] Saved marked point cloud: {marked_ply_path}")
            
            # 同时保存仅包含mask区域的点云
            mask_only_pcd = o3d.geometry.PointCloud()
            mask_only_pcd.points = o3d.utility.Vector3dVector(marked_points_np)
            mask_only_pcd.colors = o3d.utility.Vector3dVector(marked_colors_np)
            
            mask_only_path = os.path.join(output_dir, f"frame_{frame_id:06d}_mask_only.ply")
            o3d.io.write_point_cloud(mask_only_path, mask_only_pcd)
            print(f"[Info] Saved mask-only point cloud: {mask_only_path}")
        else:
            print("[Warning] No marked points collected")
            
    except Exception as e:
        print(f"[Error] Failed to process point cloud with Open3D: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理资源
    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    print("\nProcessing completed!")
    return True


def visualize_marked_point_cloud(frame_id: int, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    可视化带标记的点云
    
    Args:
        frame_id: 帧ID
        output_dir: 输出目录
    """
    marked_ply_path = os.path.join(output_dir, f"frame_{frame_id:06d}_marked.ply")
    mask_only_path = os.path.join(output_dir, f"frame_{frame_id:06d}_mask_only.ply")
    
    if not os.path.exists(marked_ply_path):
        print(f"[Error] Marked point cloud not found: {marked_ply_path}")
        return
    
    print(f"[Info] Visualizing marked point cloud...")
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(marked_ply_path)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id} - Marked Point Cloud", width=1280, height=720)
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Mark mask regions in point cloud with red color',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_point_cloud_marker.py --frame 33
    # Process frame 33 with default SVO file
    
  python mask_point_cloud_marker.py --frame 33 --svo "path\\to\\your.svo2"
    # Process frame 33 with specified SVO file
    
  python mask_point_cloud_marker.py --frame 33 --visualize
    # Process and visualize the result
        """
    )
    
    parser.add_argument(
        '--frame',
        type=int,
        required=True,
        help='Frame ID to process (e.g., 33 for frame_000033)'
    )
    
    parser.add_argument(
        '--svo',
        type=str,
        default=DEFAULT_SVO_PATH,
        help=f'Path to SVO file (default: {DEFAULT_SVO_PATH})'
    )
    
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        default=DEFAULT_SEGMENTATION_DIR,
        help=f'Segmentation output directory (default: {DEFAULT_SEGMENTATION_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the marked point cloud after processing'
    )
    
    args = parser.parse_args()
    
    # 构建mask JSON路径
    mask_json_path = os.path.join(args.segmentation_dir, 'masks', f'frame_{args.frame:06d}_masks.json')
    
    # 构建segmented图像路径
    segmented_image_path = os.path.join(args.segmentation_dir, 'segmented', f'frame_{args.frame:06d}_segmented.jpg')
    
    # 检查文件是否存在
    if not os.path.exists(mask_json_path):
        print(f"[Error] Mask JSON file not found: {mask_json_path}")
        print(f"[Info] Please check if frame {args.frame} exists in the segmentation output")
        return
    
    print(f"[Info] Found mask file: {mask_json_path}")
    
    if os.path.exists(segmented_image_path):
        print(f"[Info] Found segmented image: {segmented_image_path}")
    else:
        print(f"[Warning] Segmented image not found: {segmented_image_path}")
        print(f"[Info] Will use bbox-based mask approximation")
        segmented_image_path = None
    
    # 处理SVO文件
    success = process_svo_with_mask_marking(
        svo_path=args.svo,
        frame_id=args.frame,
        mask_json_path=mask_json_path,
        segmented_image_path=segmented_image_path,
        output_dir=args.output_dir
    )
    
    if success and args.visualize:
        visualize_marked_point_cloud(args.frame, args.output_dir)


if __name__ == "__main__":
    main()
