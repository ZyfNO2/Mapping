# -*- coding: utf-8 -*-
# SVO文件逐帧数据提取脚本
# 
# 功能：
# - 逐帧提取SVO文件中的左右目图像
# - 提取每帧的位姿信息（位置和姿态）
# - 提取相机参数
# - 保存为结构化数据
# 
# 使用方法：
#     python extract_svo_frames.py --svo path/to/your.svo2
#     python extract_svo_frames.py --svo path/to/your.svo2 --output custom_output
#     python extract_svo_frames.py --svo path/to/your.svo2 --output custom_output --skip 5
import sys
import time
import argparse
import pyzed.sl as sl
import os
import json
import numpy as np
import cv2
from pathlib import Path


def get_output_dir_from_svo(svo_path):
    """从SVO路径生成输出目录名"""
    svo_name = Path(svo_path).stem  # 获取文件名（不含扩展名）
    return f"data/extracted/{svo_name}"


def extract_camera_parameters(zed):
    """提取相机参数"""
    camera_info = zed.get_camera_information()
    
    # 左相机参数
    left_cam = camera_info.camera_configuration.calibration_parameters.left_cam
    # 右相机参数
    right_cam = camera_info.camera_configuration.calibration_parameters.right_cam
    # 基线 (手动计算，使用左右相机光心距离)
    baseline = abs(right_cam.cx - left_cam.cx) / left_cam.fx
    
    params = {
        "camera_model": camera_info.camera_model.name,
        "serial_number": camera_info.serial_number,
        "baseline": baseline,
        "left_camera": {
            "fx": left_cam.fx,
            "fy": left_cam.fy,
            "cx": left_cam.cx,
            "cy": left_cam.cy,
            "distortion": left_cam.disto.tolist()
        },
        "right_camera": {
            "fx": right_cam.fx,
            "fy": right_cam.fy,
            "cx": right_cam.cx,
            "cy": right_cam.cy,
            "distortion": right_cam.disto.tolist()
        },
        "resolution": {
            "width": camera_info.camera_configuration.resolution.width,
            "height": camera_info.camera_configuration.resolution.height
        }
    }
    
    return params


def save_image(mat, output_path):
    """保存图像"""
    image = mat.get_data()
    cv2.imwrite(output_path, image)


def save_pose(pose, output_path):
    """保存位姿信息"""
    position = pose.get_translation().get()
    orientation = pose.get_orientation().get()
    
    pose_data = {
        "position": {
            "x": position[0],
            "y": position[1],
            "z": position[2]
        },
        "orientation": {
            "x": orientation[0],
            "y": orientation[1],
            "z": orientation[2],
            "w": orientation[3]
        },
        "timestamp": pose.timestamp.data_ns
    }
    
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=2)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Extract frame-by-frame data from SVO file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python extract_svo_frames.py --svo "path\to\your.svo2"
    # Extract data from specified SVO file
    
  python extract_svo_frames.py --svo "path\to\your.svo2" --output "custom_folder"
    # Specify custom output directory
    
  python extract_svo_frames.py --svo "path\to\your.svo2" --skip 5
    # Skip every 5 frames
        """
    )
    
    parser.add_argument(
        '--svo',
        type=str,
        required=True,
        help='Path to SVO file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data/extracted/<svo_filename>)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Number of frames to skip between extractions (default: 0)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='Maximum number of frames to extract (default: all)'
    )
    
    args = parser.parse_args()
    
    # 获取SVO路径
    svo_path = args.svo
    
    # 检查SVO文件是否存在
    if not Path(svo_path).exists():
        print(f"Error: SVO file not found: {svo_path}")
        return
    
    # 获取输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = get_output_dir_from_svo(svo_path)
    
    # 创建输出目录结构
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "left")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "right")).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, "pose")).mkdir(exist_ok=True)
    
    print("="*70)
    print("SVO Frame-by-Frame Data Extraction")
    print("="*70)
    print(f"SVO file: {svo_path}")
    print(f"Output directory: {output_dir}")
    print(f"Skip frames: {args.skip}")
    print(f"Max frames: {args.max_frames if args.max_frames > 0 else 'all'}")
    print("="*70)
    
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # SVO文件路径
    init.set_from_svo_file(svo_path)
    
    # 打开相机
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {repr(status)}. Exit program.")
        return
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"Enable Positional Tracking : {repr(returned_state)}. Exit program.")
        zed.close()
        return
    
    # 提取相机参数
    camera_params = extract_camera_parameters(zed)
    with open(os.path.join(output_dir, "camera_parameters.json"), 'w') as f:
        json.dump(camera_params, f, indent=2)
    print("Saved camera parameters to camera_parameters.json")
    
    # 准备数据容器
    left_image = sl.Mat()
    right_image = sl.Mat()
    pose = sl.Pose()
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    # 主循环
    frame_count = 0
    extracted_count = 0
    
    print("Processing SVO file...")
    
    while True:
        grab_result = zed.grab(runtime_parameters)
        
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed.")
            break
        elif grab_result != sl.ERROR_CODE.SUCCESS:
            print(f"Error during grab: {grab_result}")
            continue
        
        # 检查是否需要跳过当前帧
        if frame_count % (args.skip + 1) != 0:
            frame_count += 1
            continue
        
        # 检查是否达到最大帧数
        if args.max_frames > 0 and extracted_count >= args.max_frames:
            print(f"Reached maximum frames: {args.max_frames}")
            break
        
        # 提取左目图像
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        left_output_path = os.path.join(output_dir, "left", f"frame_{extracted_count:06d}.png")
        save_image(left_image, left_output_path)
        
        # 提取右目图像
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        right_output_path = os.path.join(output_dir, "right", f"frame_{extracted_count:06d}.png")
        save_image(right_image, right_output_path)
        
        # 提取位姿信息
        zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        pose_output_path = os.path.join(output_dir, "pose", f"frame_{extracted_count:06d}.json")
        save_pose(pose, pose_output_path)
        
        # 显示进度
        if extracted_count % 10 == 0:
            print(f"Extracted frame {extracted_count}")
        
        extracted_count += 1
        frame_count += 1
    
    # 关闭相机
    zed.close()
    
    # 显示结果
    print("\n" + "="*70)
    print("Extraction completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames extracted: {extracted_count}")
    print(f"Output files in: {output_dir}")
    print("\nExtracted data:")
    print(f"  left/          - Left camera images")
    print(f"  right/         - Right camera images")
    print(f"  pose/          - Pose information (JSON)")
    print(f"  camera_parameters.json - Camera calibration data")
    print("="*70)


if __name__ == "__main__":
    main()
