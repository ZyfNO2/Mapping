# -*- coding: utf-8 -*-
"""
SVO分段处理测试脚本 - 测试分段处理SVO文件的可行性

功能：
1. 将SVO视频文件分割为前半段和后半段两个部分
2. 分别处理这两个分段，生成点云数据
3. 将前半段和后半段生成的点云保存为独立的PLY文件
4. 实现将两个PLY文件组合成一个完整点云
5. 对原始完整SVO文件进行完整处理，生成完整的点云PLY文件
6. 共生成四个点云文件用于对比分析

    python test_svo_cut.py --svo "path/to/your.svo2"
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
from datetime import datetime
import logging


DEFAULT_SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-17-12.svo2"
DEFAULT_MODEL_PATH = r"G:\Zed\spatial mapping\python\seg.pt"
DEFAULT_OUTPUT_DIR = "data/svo_cut_test"


class SVOCutLogger:
    """日志记录器类"""

    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)


def get_svo_total_frames(svo_path: str) -> int:
    """
    获取SVO文件的总帧数

    Args:
        svo_path: SVO文件路径

    Returns:
        总帧数
    """
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")

    total_frames = zed.get_svo_number_of_frames()
    zed.close()
    return total_frames


def process_svo_segment(
    svo_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    logger: SVOCutLogger,
    spatial_mapping_params: sl.SpatialMappingParameters = None,
    segment_name: str = "",
    initial_pose: Tuple[np.ndarray, np.ndarray] = None
) -> Tuple[bool, Optional[Dict]]:
    """
    处理SVO文件的指定分段

    Args:
        svo_path: SVO文件路径
        start_frame: 起始帧
        end_frame: 结束帧
        output_path: 输出PLY文件路径
        logger: 日志记录器
        spatial_mapping_params: 空间映射参数
        segment_name: 分段名称标识
        initial_pose: 初始位姿 (rotation, translation) 元组，用于位姿连续性

    Returns:
        是否成功, 位姿信息字典
    """
    pose_info = {
        "segment_name": segment_name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_position": None,
        "start_rotation": None,
        "end_position": None,
        "end_rotation": None
    }

    logger.info(f"Processing segment: frames {start_frame} to {end_frame}")

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    init.set_from_svo_file(svo_path)

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        logger.error(f"Camera Open: {repr(status)}")
        return False

    zed.set_svo_position(start_frame)
    logger.info(f"Set SVO position to frame {start_frame}")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True

    if initial_pose is not None:
        from scipy.spatial.transform import Rotation
        rot, trans = initial_pose
        scipy_rot = Rotation.from_matrix(rot)
        quat = scipy_rot.as_quat()
        orient = sl.Orientation()
        orient.init_vector(quat[0], quat[1], quat[2], quat[3])
        translation = sl.Translation()
        translation.init_vector(trans[0], trans[1], trans[2])
        transform = sl.Transform()
        transform.init_orientation_translation(orient, translation)
        positional_tracking_parameters.set_initial_world_transform(transform)
        logger.info(f"[{segment_name}] Set initial pose from previous segment")

    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Enable Positional Tracking: {repr(returned_state)}")
        zed.close()
        return False, pose_info

    if spatial_mapping_params is None:
        spatial_mapping_params = sl.SpatialMappingParameters(
            resolution=sl.MAPPING_RESOLUTION.HIGH,
            mapping_range=sl.MAPPING_RANGE.SHORT,
            max_memory_usage=8192,
            save_texture=True,
            use_chunk_only=False,
            reverse_vertex_order=False,
            map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        )
        spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
        spatial_mapping_params.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.SHORT)
        spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

    zed.enable_spatial_mapping(spatial_mapping_params)

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50

    pose = sl.Pose()
    frame_count = 0
    target_frames = end_frame - start_frame
    first_pose_captured = False

    logger.info(f"Starting frame processing, target {target_frames} frames...")

    while frame_count < target_frames:
        grab_result = zed.grab(runtime_parameters)

        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            logger.warning("Reached end of SVO file unexpectedly")
            break

        if grab_result <= sl.ERROR_CODE.SUCCESS:
            zed.get_position(pose)

            if not first_pose_captured and frame_count == 0:
                pose_data = pose.pose_data()
                translation = np.array([pose_data[0, 3], pose_data[1, 3], pose_data[2, 3]])
                rotation = np.array([
                    [pose_data[0, 0], pose_data[0, 1], pose_data[0, 2]],
                    [pose_data[1, 0], pose_data[1, 1], pose_data[1, 2]],
                    [pose_data[2, 0], pose_data[2, 1], pose_data[2, 2]]
                ])
                pose_info["start_position"] = translation
                pose_info["start_rotation"] = rotation
                logger.info(f"[{segment_name}] Start Position: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
                logger.info(f"[{segment_name}] Start Rotation (3x3 matrix):\n{rotation}")
                first_pose_captured = True

            if frame_count == target_frames - 1:
                pose_data = pose.pose_data()
                translation = np.array([pose_data[0, 3], pose_data[1, 3], pose_data[2, 3]])
                rotation = np.array([
                    [pose_data[0, 0], pose_data[0, 1], pose_data[0, 2]],
                    [pose_data[1, 0], pose_data[1, 1], pose_data[1, 2]],
                    [pose_data[2, 0], pose_data[2, 1], pose_data[2, 2]]
                ])
                pose_info["end_position"] = translation
                pose_info["end_rotation"] = rotation
                logger.info(f"[{segment_name}] End Position: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
                logger.info(f"[{segment_name}] End Rotation (3x3 matrix):\n{rotation}")

            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{target_frames} frames")

        if frame_count >= target_frames:
            break

    logger.info(f"Frame capture completed, {frame_count} frames processed")

    logger.info("Extracting spatial map...")
    point_cloud = sl.FusedPointCloud()
    extract_status = zed.extract_whole_spatial_map(point_cloud)

    if extract_status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Failed to extract point cloud: {extract_status}")
        zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
        return False, pose_info

    logger.info(f"Saving point cloud to: {output_path}")
    save_status = point_cloud.save(output_path, sl.MESH_FILE_FORMAT.PLY)

    if not save_status:
        logger.error(f"Failed to save PLY: {save_status}")
        point_cloud.clear()
        zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
        return False, pose_info

    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

    logger.info(f"Segment processing completed: {output_path}")
    return True, pose_info


def process_full_svo(
    svo_path: str,
    output_path: str,
    logger: SVOCutLogger,
    spatial_mapping_params: sl.SpatialMappingParameters = None
) -> bool:
    """
    处理完整的SVO文件

    Args:
        svo_path: SVO文件路径
        output_path: 输出PLY文件路径
        logger: 日志记录器
        spatial_mapping_params: 空间映射参数

    Returns:
        是否成功
    """
    logger.info("Processing full SVO file...")

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    init.set_from_svo_file(svo_path)

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        logger.error(f"Camera Open: {repr(status)}")
        return False

    camera_info = zed.get_camera_information()
    total_frames = zed.get_svo_number_of_frames()
    logger.info(f"Camera resolution: {camera_info.camera_configuration.resolution.width}x{camera_info.camera_configuration.resolution.height}")
    logger.info(f"Total frames in SVO: {total_frames}")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Enable Positional Tracking: {repr(returned_state)}")
        zed.close()
        return False

    if spatial_mapping_params is None:
        spatial_mapping_params = sl.SpatialMappingParameters(
            resolution=sl.MAPPING_RESOLUTION.HIGH,
            mapping_range=sl.MAPPING_RANGE.SHORT,
            max_memory_usage=8192,
            save_texture=True,
            use_chunk_only=False,
            reverse_vertex_order=False,
            map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        )
        spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
        spatial_mapping_params.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.SHORT)
        spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

    zed.enable_spatial_mapping(spatial_mapping_params)

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50

    frame_count = 0

    logger.info("Starting full SVO frame processing...")

    while True:
        grab_result = zed.grab(runtime_parameters)

        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            logger.info("SVO file playback completed")
            break

        if grab_result <= sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

    logger.info(f"Frame capture completed, {frame_count} frames processed")

    logger.info("Extracting spatial map...")
    point_cloud = sl.FusedPointCloud()
    extract_status = zed.extract_whole_spatial_map(point_cloud)

    if extract_status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"Failed to extract point cloud: {extract_status}")
        zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
        return False

    logger.info(f"Saving point cloud to: {output_path}")
    save_status = point_cloud.save(output_path, sl.MESH_FILE_FORMAT.PLY)

    if not save_status:
        logger.error(f"Failed to save PLY: {save_status}")
        point_cloud.clear()
        zed.disable_spatial_mapping()
        zed.disable_positional_tracking()
        zed.close()
        return False

    point_cloud.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()

    logger.info(f"Full SVO processing completed: {output_path}")
    return True


def combine_ply_files(
    ply_files: List[str],
    output_path: str,
    logger: SVOCutLogger
) -> bool:
    """
    合并多个PLY文件为一个点云

    Args:
        ply_files: PLY文件路径列表
        output_path: 输出文件路径
        logger: 日志记录器

    Returns:
        是否成功
    """
    logger.info(f"Combining {len(ply_files)} PLY files...")

    combined_points = []
    combined_colors = []

    for ply_file in ply_files:
        if not os.path.exists(ply_file):
            logger.warning(f"PLY file not found: {ply_file}, skipping")
            continue

        logger.info(f"Loading: {ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.7

        combined_points.append(points)
        combined_colors.append(colors)

        logger.info(f"  Loaded {len(points)} points")

    if len(combined_points) == 0:
        logger.error("No PLY files to combine")
        return False

    all_points = np.vstack(combined_points)
    all_colors = np.vstack(combined_colors)

    logger.info(f"Combined point cloud: {len(all_points)} points")

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    logger.info(f"Saving combined point cloud to: {output_path}")
    save_status = o3d.io.write_point_cloud(output_path, combined_pcd)

    if not save_status:
        logger.error("Failed to save combined PLY")
        return False

    logger.info(f"Combined PLY saved successfully: {output_path}")
    return True


def test_svo_cut_process(
    svo_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    log_file: str = None
):
    """
    测试SVO分段处理的主函数

    Args:
        svo_path: SVO文件路径
        output_dir: 输出目录
        log_file: 日志文件路径
    """
    logger = SVOCutLogger("SVOCutTest", log_file)

    logger.info("=" * 70)
    logger.info("SVO分段处理测试开始")
    logger.info("=" * 70)
    logger.info(f"SVO文件: {svo_path}")
    logger.info(f"输出目录: {output_dir}")

    if not os.path.exists(svo_path):
        logger.error(f"SVO文件不存在: {svo_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    svo_name = Path(svo_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("获取SVO文件总帧数...")
    total_frames = get_svo_total_frames(svo_path)
    logger.info(f"SVO总帧数: {total_frames}")

    mid_frame = total_frames // 2
    logger.info(f"分割点: 帧 {mid_frame} (总帧数: {total_frames})")

    first_half_ply = os.path.join(output_dir, f"{svo_name}_first_half_{timestamp}.ply")
    second_half_ply = os.path.join(output_dir, f"{svo_name}_second_half_{timestamp}.ply")
    combined_ply = os.path.join(output_dir, f"{svo_name}_combined_{timestamp}.ply")
    full_ply = os.path.join(output_dir, f"{svo_name}_full_{timestamp}.ply")

    spatial_mapping_params = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.HIGH,
        mapping_range=sl.MAPPING_RANGE.SHORT,
        max_memory_usage=8192,
        save_texture=True,
        use_chunk_only=False,
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    )
    spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
    spatial_mapping_params.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.SHORT)
    spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

    phase1_start = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("阶段1: 处理前半段SVO")
    logger.info("=" * 70)
    phase1_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"开始时间: {phase1_start_time}")

    success1, pose1 = process_svo_segment(
        svo_path=svo_path,
        start_frame=0,
        end_frame=mid_frame,
        output_path=first_half_ply,
        logger=logger,
        spatial_mapping_params=spatial_mapping_params,
        segment_name="First_Half"
    )

    phase1_end = time.time()
    phase1_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"结束时间: {phase1_end_time}")
    logger.info(f"耗时: {phase1_end - phase1_start:.2f} 秒")

    if not success1:
        logger.error("前半段处理失败，终止测试")
        return False

    phase2_start = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("阶段2: 处理后半段SVO")
    logger.info("=" * 70)
    phase2_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"开始时间: {phase2_start_time}")

    success2, pose2 = process_svo_segment(
        svo_path=svo_path,
        start_frame=mid_frame,
        end_frame=total_frames,
        output_path=second_half_ply,
        logger=logger,
        spatial_mapping_params=spatial_mapping_params,
        segment_name="Second_Half",
        initial_pose=(pose1["end_rotation"], pose1["end_position"])
    )

    phase2_end = time.time()
    phase2_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"结束时间: {phase2_end_time}")
    logger.info(f"耗时: {phase2_end - phase2_start:.2f} 秒")

    if not success2:
        logger.error("后半段处理失败，终止测试")
        return False

    phase3_start = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("阶段3: 合并前后两段点云")
    logger.info("=" * 70)
    phase3_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"开始时间: {phase3_start_time}")

    success3 = combine_ply_files(
        ply_files=[first_half_ply, second_half_ply],
        output_path=combined_ply,
        logger=logger
    )

    phase3_end = time.time()
    phase3_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"结束时间: {phase3_end_time}")
    logger.info(f"耗时: {phase3_end - phase3_start:.2f} 秒")

    if not success3:
        logger.error("点云合并失败，终止测试")
        return False

    logger.info("")
    logger.info("=" * 70)
    logger.info("位姿对比分析")
    logger.info("=" * 70)
    logger.info("")
    logger.info("前半段终止位姿 vs 后半段起始位姿:")
    logger.info("")

    if pose1 and pose2:
        pos1_end = pose1["end_position"]
        pos2_start = pose2["start_position"]
        rot1_end = pose1["end_rotation"]
        rot2_start = pose2["start_rotation"]

        if pos1_end is not None and pos2_start is not None:
            pos_diff = pos2_start - pos1_end
            pos_distance = np.linalg.norm(pos_diff)
            logger.info(f"后半段起始位置 - 前半段终止位置 = [{pos_diff[0]:.4f}, {pos_diff[1]:.4f}, {pos_diff[2]:.4f}]")
            logger.info(f"位置偏差距离: {pos_distance:.4f} 米")

        if rot1_end is not None and rot2_start is not None:
            rot_diff = rot2_start @ rot1_end.T
            trace = np.trace(rot_diff)
            angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            angle_deg = np.degrees(angle_rad)
            logger.info(f"旋转偏差角度: {angle_deg:.4f} 度")

    logger.info("")

    phase4_start = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("阶段4: 完整SVO处理")
    logger.info("=" * 70)
    phase4_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"开始时间: {phase4_start_time}")

    success4 = process_full_svo(
        svo_path=svo_path,
        output_path=full_ply,
        logger=logger,
        spatial_mapping_params=spatial_mapping_params
    )

    phase4_end = time.time()
    phase4_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"结束时间: {phase4_end_time}")
    logger.info(f"耗时: {phase4_end - phase4_start:.2f} 秒")

    if not success4:
        logger.error("完整SVO处理失败")
        return False

    logger.info("")
    logger.info("=" * 70)
    logger.info("测试完成 - 结果汇总")
    logger.info("=" * 70)
    logger.info("")
    logger.info("生成的点云文件:")
    logger.info(f"  1. 前半段PLY: {first_half_ply}")
    logger.info(f"  2. 后半段PLY: {second_half_ply}")
    logger.info(f"  3. 组合PLY:   {combined_ply}")
    logger.info(f"  4. 完整PLY:   {full_ply}")
    logger.info("")
    logger.info("处理耗时:")
    logger.info(f"  前半段处理: {phase1_end - phase1_start:.2f} 秒")
    logger.info(f"  后半段处理: {phase2_end - phase2_start:.2f} 秒")
    logger.info(f"  点云合并:   {phase3_end - phase3_start:.2f} 秒")
    logger.info(f"  完整处理:   {phase4_end - phase4_start:.2f} 秒")
    logger.info(f"  总耗时:     {phase4_end - phase1_start:.2f} 秒")
    logger.info("")

    if os.path.exists(first_half_ply):
        pcd1 = o3d.io.read_point_cloud(first_half_ply)
        logger.info(f"前半段点云点数: {len(pcd1.points)}")

    if os.path.exists(second_half_ply):
        pcd2 = o3d.io.read_point_cloud(second_half_ply)
        logger.info(f"后半段点云点数: {len(pcd2.points)}")

    if os.path.exists(combined_ply):
        pcd_combined = o3d.io.read_point_cloud(combined_ply)
        logger.info(f"组合点云点数: {len(pcd_combined.points)}")

    if os.path.exists(full_ply):
        pcd_full = o3d.io.read_point_cloud(full_ply)
        logger.info(f"完整点云点数: {len(pcd_full.points)}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SVO分段处理测试完成")
    logger.info("=" * 70)

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SVO分段处理测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_svo_cut.py --svo "path/to/your.svo2"
  python test_svo_cut.py --svo "path/to/your.svo2" --output-dir "data/test_output"
        """
    )

    parser.add_argument(
        '--svo',
        type=str,
        default=DEFAULT_SVO_PATH,
        help=f'SVO文件路径 (默认: {DEFAULT_SVO_PATH})'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--log',
        type=str,
        default=None,
        help='日志文件路径'
    )

    args = parser.parse_args()

    if not os.path.exists(args.svo):
        print(f"[Error] SVO文件不存在: {args.svo}")
        return

    test_svo_cut_process(
        svo_path=args.svo,
        output_dir=args.output_dir,
        log_file=args.log
    )


if __name__ == "__main__":
    main()
