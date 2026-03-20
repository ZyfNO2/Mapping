# -*- coding: utf-8 -*-
"""
SVO损伤检测与点云标记一体化脚本

功能：
1. 读取SVO文件进行空间映射重建
2. 实时运行YOLO损伤检测（best.pt）
3. 将检测到的损伤区域点云标记为红色
4. 生成三种点云文件：
   - {svo_name}_original.ply: 原始重建点云（无标记）
   - {svo_name}_marked.ply: 带红色标记的完整点云
   - {svo_name}_damage_only.ply: 仅包含损伤区域的点云

使用方法：
    python svo_damage_detection.py --svo "path/to/your.svo2"
    python svo_damage_detection.py --svo "path/to/your.svo2" --model "path/to/best.pt"
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
    
    Args:
        zed: ZED相机对象
        mask: 二值mask
        pose: 当前帧位姿
        sample_step: 采样步长
        
    Returns:
        damage_points: 损伤区域的3D点列表（世界坐标系）
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


def process_svo_with_damage_detection(
    svo_path: str,
    model_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    conf_threshold: float = CONF_THRESHOLD,
    save_detection_images: bool = True
):
    """
    处理SVO文件，进行损伤检测并标记点云
    
    Args:
        svo_path: SVO文件路径
        model_path: YOLO模型路径
        output_dir: 输出目录
        conf_threshold: 检测置信度阈值
        save_detection_images: 是否保存检测图像
    """
    print("="*70)
    print("SVO Damage Detection & Point Cloud Marking")
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
                    # 绘制检测结果
                    result_image = frame_bgr.copy()
                    for mask, conf in zip(masks, confidences):
                        # 创建彩色mask
                        mask_bool = mask > 0.5
                        colored_mask = np.zeros_like(result_image)
                        colored_mask[mask_bool] = [0, 0, 255]  # 红色
                        
                        # 叠加
                        result_image[mask_bool] = cv2.addWeighted(
                            result_image[mask_bool], 0.5,
                            colored_mask[mask_bool], 0.5, 0
                        )
                    
                    # 保存图像
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
            
            # 查找并标记
            threshold = 0.05  # 5cm阈值
            marked_count = 0
            
            for i, pt in enumerate(original_points):
                dist, idx = damage_tree.query(pt, distance_upper_bound=threshold)
                if dist != np.inf:
                    original_colors[i] = MARK_COLOR
                    marked_count += 1
            
            print(f"[Info] Marked {marked_count} points as damage regions")
            
            # 保存带标记的点云
            pcd.colors = o3d.utility.Vector3dVector(original_colors)
            marked_ply_path = os.path.join(output_dir, f"{svo_name}_marked.ply")
            o3d.io.write_point_cloud(marked_ply_path, pcd)
            print(f"[Info] Saved marked point cloud: {marked_ply_path}")
            
            # 保存仅损伤区域的点云
            damage_only_pcd = o3d.geometry.PointCloud()
            damage_only_pcd.points = o3d.utility.Vector3dVector(damage_points_np)
            damage_colors = np.ones((len(damage_points_np), 3)) * MARK_COLOR
            damage_only_pcd.colors = o3d.utility.Vector3dVector(damage_colors)
            
            damage_only_path = os.path.join(output_dir, f"{svo_name}_damage_only.ply")
            o3d.io.write_point_cloud(damage_only_path, damage_only_pcd)
            print(f"[Info] Saved damage-only point cloud: {damage_only_path}")
        else:
            print("[Warning] No damage points detected")
            
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
    print("="*70)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SVO Damage Detection & Point Cloud Marking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python svo_damage_detection.py --svo "path\\to\\your.svo2"
  python svo_damage_detection.py --svo "path\\to\\your.svo2" --model "path\\to\\best.pt"
  python svo_damage_detection.py --svo "path\\to\\your.svo2" --conf 0.2
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
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.svo):
        print(f"[Error] SVO file not found: {args.svo}")
        return
    
    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        return
    
    # 运行处理
    process_svo_with_damage_detection(
        svo_path=args.svo,
        model_path=args.model,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        save_detection_images=not args.no_images
    )


if __name__ == "__main__":
    main()
