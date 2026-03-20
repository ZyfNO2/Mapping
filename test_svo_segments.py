# -*- coding: utf-8 -*-
"""
SVO分段测试脚本
用于分析SVO文件中点云重复重建问题

功能：
1. 截取SVO的前1/2进行测试
2. 截取SVO的前3/4进行测试
3. 对比分析点云重建差异

使用方法：
    python test_svo_segments.py --svo "path/to/your.svo2"
    python test_svo_segments.py --svo "path/to/your.svo2" --model "path/to/best.pt"
"""

import sys
import argparse
import pyzed.sl as sl
import os
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
from collections import defaultdict
import shutil


# 默认SVO文件路径
DEFAULT_SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-18-11.svo2"

# 默认模型路径
DEFAULT_MODEL_PATH = r"G:\Zed\spatial mapping\python\seg.pt"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "data/segment_tests"

# 检测置信度阈值
CONF_THRESHOLD = 0.15

# 标记颜色（红色）
MARK_COLOR = [1.0, 0.0, 0.0]


class DamageDetector:
    """损伤检测器类"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.15):
        print(f"[Info] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("[Info] Model loaded successfully!")
        
    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, verbose=False, conf=self.conf_threshold)
        
        masks = []
        confidences = []
        boxes = []
        
        if results[0].masks is not None and len(results[0].masks) > 0:
            result_masks = results[0].masks.data.cpu().numpy()
            result_boxes = results[0].boxes.xyxy.cpu().numpy()
            result_confs = results[0].boxes.conf.cpu().numpy()
            
            for mask, box, conf in zip(result_masks, result_boxes, result_confs):
                if conf >= self.conf_threshold:
                    mask_resized = cv2.resize(
                        mask, 
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    masks.append(mask_resized)
                    confidences.append(float(conf))
                    boxes.append(box)
        
        return masks, confidences, boxes


def extract_damage_points_from_frame(zed: sl.Camera, mask: np.ndarray, pose: sl.Pose, sample_step: int = 5) -> List[np.ndarray]:
    """从单帧中提取损伤区域的3D点云"""
    point_cloud_frame = sl.Mat()
    zed.retrieve_measure(point_cloud_frame, sl.MEASURE.XYZRGBA)
    
    mask_coords = np.where(mask > 0.5)
    mask_y_coords = mask_coords[0]
    mask_x_coords = mask_coords[1]
    
    pose_data = pose.pose_data()
    rotation = np.array([
        [pose_data[0, 0], pose_data[0, 1], pose_data[0, 2]],
        [pose_data[1, 0], pose_data[1, 1], pose_data[1, 2]],
        [pose_data[2, 0], pose_data[2, 1], pose_data[2, 2]]
    ])
    translation = np.array([pose_data[0, 3], pose_data[1, 3], pose_data[2, 3]])
    
    damage_points = []
    
    for i in range(0, len(mask_y_coords), sample_step):
        y = mask_y_coords[i]
        x = mask_x_coords[i]
        
        point_3d = point_cloud_frame.get_value(x, y)[1]
        
        if (not np.isnan(point_3d[0]) and 
            not np.isnan(point_3d[1]) and 
            not np.isnan(point_3d[2]) and
            abs(point_3d[0]) < 10 and 
            abs(point_3d[1]) < 10 and 
            abs(point_3d[2]) < 10):
            
            point_camera = np.array([point_3d[0], point_3d[1], point_3d[2]])
            point_world = rotation @ point_camera + translation
            damage_points.append(point_world)
    
    point_cloud_frame.free(sl.MEM.CPU)
    return damage_points


def process_svo_segment(
    svo_path: str,
    model_path: str,
    output_dir: str,
    segment_name: str,
    max_frames: int = None,
    conf_threshold: float = CONF_THRESHOLD
):
    """
    处理SVO的指定帧数片段
    
    Args:
        svo_path: SVO文件路径
        model_path: 模型路径
        output_dir: 输出目录
        segment_name: 片段名称（如"first_half", "first_three_quarters"）
        max_frames: 最大处理帧数（None表示处理全部）
        conf_threshold: 置信度阈值
    """
    print("\n" + "="*70)
    print(f"Processing SVO Segment: {segment_name}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print("="*70)
    
    # 获取SVO文件名
    svo_name = Path(svo_path).stem
    segment_output_dir = os.path.join(output_dir, f"{svo_name}_{segment_name}")
    os.makedirs(segment_output_dir, exist_ok=True)
    
    print(f"[Info] Output directory: {segment_output_dir}")
    
    # 初始化检测器
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
        return None
    
    # 获取总帧数
    total_frames = zed.get_svo_number_of_frames()
    print(f"[Info] Total SVO frames: {total_frames}")
    
    if max_frames is None:
        max_frames = total_frames
    else:
        max_frames = min(max_frames, total_frames)
    
    print(f"[Info] Will process first {max_frames} frames")
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Enable Positional Tracking: {repr(returned_state)}. Exit program.")
        zed.close()
        return None
    
    # 配置空间映射
    spatial_mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.HIGH,
        mapping_range=sl.MAPPING_RANGE.SHORT,
        max_memory_usage=6144,
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
    
    # 统计数据
    all_damage_points = []
    damage_frame_count = 0
    total_detections = 0
    frame_count = 0
    
    print("\n[Info] Processing frames...")
    
    while frame_count < max_frames:
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
                
                if frame_count % 10 == 0 or frame_count < 20:
                    print(f"[Frame {frame_count}] Detected {len(masks)} damage regions")
                
                # 提取每个mask区域的点云
                for i, (mask, conf, box) in enumerate(zip(masks, confidences, boxes)):
                    damage_points = extract_damage_points_from_frame(zed, mask, pose)
                    
                    if len(damage_points) > 0:
                        all_damage_points.extend(damage_points)
                        if frame_count % 10 == 0 or frame_count < 20:
                            print(f"  - Damage {i+1}: {len(damage_points)} points (conf: {conf:.3f})")
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                progress = (frame_count / max_frames) * 100
                print(f"[Progress] {frame_count}/{max_frames} frames ({progress:.1f}%)")
    
    print(f"\n[Info] Detection summary for {segment_name}:")
    print(f"  - Processed frames: {frame_count}")
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
        return None
    
    # 保存原始点云
    original_ply_path = os.path.join(segment_output_dir, f"{svo_name}_{segment_name}_original.ply")
    point_cloud.save(original_ply_path, sl.MESH_FILE_FORMAT.PLY)
    
    # 读取并处理点云
    try:
        pcd = o3d.io.read_point_cloud(original_ply_path)
        original_points = np.asarray(pcd.points)
        
        print(f"[Info] Full point cloud has {len(original_points)} points")
        
        # 创建标记点云
        marked_pcd = o3d.geometry.PointCloud()
        marked_pcd.points = pcd.points
        
        if len(all_damage_points) > 0:
            damage_points_np = np.array(all_damage_points)
            
            # 创建颜色数组
            if pcd.has_colors():
                original_colors = np.asarray(pcd.colors)
            else:
                original_colors = np.ones((len(original_points), 3)) * 0.7
            
            # 使用KDTree匹配损伤点
            print("[Info] Matching damage points...")
            from scipy.spatial import cKDTree
            damage_tree = cKDTree(damage_points_np)
            
            threshold = 0.05
            marked_count = 0
            
            for i, pt in enumerate(original_points):
                dist, idx = damage_tree.query(pt, distance_upper_bound=threshold)
                if dist != np.inf:
                    original_colors[i] = MARK_COLOR
                    marked_count += 1
            
            print(f"[Info] Marked {marked_count} points as damage regions")
            marked_pcd.colors = o3d.utility.Vector3dVector(original_colors)
            
            # 保存仅损伤区域的点云
            damage_only_pcd = o3d.geometry.PointCloud()
            damage_only_pcd.points = o3d.utility.Vector3dVector(damage_points_np)
            damage_colors = np.ones((len(damage_points_np), 3)) * MARK_COLOR
            damage_only_pcd.colors = o3d.utility.Vector3dVector(damage_colors)
            
            damage_only_path = os.path.join(segment_output_dir, f"{svo_name}_{segment_name}_damage_only.ply")
            o3d.io.write_point_cloud(damage_only_path, damage_only_pcd)
        else:
            print("[Warning] No damage points detected")
            marked_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(original_points), 3)) * 0.7)
        
        # 保存标记点云
        marked_ply_path = os.path.join(segment_output_dir, f"{svo_name}_{segment_name}_marked.ply")
        o3d.io.write_point_cloud(marked_ply_path, marked_pcd)
        
    except Exception as e:
        print(f"[Error] Failed to process point cloud: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    print(f"\n[Info] Segment {segment_name} completed!")
    print(f"[Info] Output: {segment_output_dir}")
    
    return {
        'segment_name': segment_name,
        'total_frames': frame_count,
        'damage_frames': damage_frame_count,
        'total_detections': total_detections,
        'damage_points': len(all_damage_points),
        'output_dir': segment_output_dir,
        'point_cloud_points': len(original_points) if 'original_points' in locals() else 0
    }


def compare_segments(results: list):
    """对比不同片段的结果"""
    print("\n" + "="*70)
    print("Segment Comparison Analysis")
    print("="*70)
    
    if len(results) < 2:
        print("[Warning] Need at least 2 segments to compare")
        return
    
    # 打印对比表格
    print("\n{:<25} {:<12} {:<12} {:<15} {:<15}".format(
        "Segment", "Frames", "Damage Frames", "Damage Points", "Point Cloud"
    ))
    print("-" * 70)
    
    for r in results:
        print("{:<25} {:<12} {:<12} {:<15} {:<15}".format(
            r['segment_name'],
            r['total_frames'],
            r['damage_frames'],
            r['damage_points'],
            r['point_cloud_points']
        ))
    
    # 分析重复重建问题
    print("\n[Analysis] Checking for duplicate reconstruction...")
    
    if len(results) >= 2:
        first = results[0]
        second = results[1]
        
        # 计算点云增长比例
        if first['point_cloud_points'] > 0:
            growth_ratio = second['point_cloud_points'] / first['point_cloud_points']
            print(f"\n[Analysis] Point cloud growth from {first['segment_name']} to {second['segment_name']}:")
            print(f"  - Ratio: {growth_ratio:.2f}x")
            print(f"  - Expected ratio (frame count): {second['total_frames'] / first['total_frames']:.2f}x")
            
            if growth_ratio > (second['total_frames'] / first['total_frames']) * 1.5:
                print("  - [WARNING] Significant duplicate reconstruction detected!")
                print(f"    Actual growth ({growth_ratio:.2f}x) is much higher than expected ({second['total_frames'] / first['total_frames']:.2f}x)")
            else:
                print("  - [OK] Point cloud growth is within expected range")
    
    print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SVO Segment Test - Analyze duplicate reconstruction issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_svo_segments.py --svo "path\\to\\your.svo2"
  python test_svo_segments.py --svo "path\\to\\your.svo2" --model "path\\to\\best.pt"
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
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.svo):
        print(f"[Error] SVO file not found: {args.svo}")
        return
    
    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        return
    
    # 获取总帧数
    print("[Info] Getting SVO info...")
    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Cannot open SVO: {status}")
        return
    
    total_frames = zed.get_svo_number_of_frames()
    zed.close()
    
    print(f"[Info] Total SVO frames: {total_frames}")
    
    # 计算分割点
    half_frame = total_frames // 2
    three_quarter_frame = int(total_frames * 0.75)
    
    print(f"[Info] First 1/2: frames 0-{half_frame}")
    print(f"[Info] First 3/4: frames 0-{three_quarter_frame}")
    print(f"[Info] Full: frames 0-{total_frames}")
    
    # 运行三个测试
    results = []
    
    # 1. 前1/2
    result_half = process_svo_segment(
        svo_path=args.svo,
        model_path=args.model,
        output_dir=args.output_dir,
        segment_name="first_half",
        max_frames=half_frame,
        conf_threshold=args.conf
    )
    if result_half:
        results.append(result_half)
    
    # 2. 前3/4
    result_three_quarters = process_svo_segment(
        svo_path=args.svo,
        model_path=args.model,
        output_dir=args.output_dir,
        segment_name="first_three_quarters",
        max_frames=three_quarter_frame,
        conf_threshold=args.conf
    )
    if result_three_quarters:
        results.append(result_three_quarters)
    
    # 3. 完整（可选，如果不需要可以注释掉）
    # result_full = process_svo_segment(
    #     svo_path=args.svo,
    #     model_path=args.model,
    #     output_dir=args.output_dir,
    #     segment_name="full",
    #     max_frames=None,
    #     conf_threshold=args.conf
    # )
    # if result_full:
    #     results.append(result_full)
    
    # 对比分析
    compare_segments(results)
    
    print("\n" + "="*70)
    print("All segment tests completed!")
    print(f"Output directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
