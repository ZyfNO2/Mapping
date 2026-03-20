# -*- coding: utf-8 -*-
"""
批量生成Mesh文件脚本
使用ZED SpatialMapping处理SVO文件生成mesh

功能：
1. 读取指定目录下的所有SVO文件
2. 使用SpatialMapping生成mesh（.obj格式）
3. 输出到与damage_detection相同的文件夹结构

使用方法：
    python batch_generate_mesh.py
    python batch_generate_mesh.py --svo-dir "C:/Users/ZYF/Documents/ZED"
    python batch_generate_mesh.py --output-dir "data/damage_detection"
"""

import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime
import pyzed.sl as sl
import numpy as np


# 默认配置
DEFAULT_SVO_DIR = r"C:\Users\ZYF\Documents\ZED"
DEFAULT_OUTPUT_DIR = "data/damage_detection"


def generate_mesh_from_svo(svo_path: str, output_dir: str) -> bool:
    """
    从SVO文件生成mesh
    
    Args:
        svo_path: SVO文件路径
        output_dir: 输出目录
        
    Returns:
        bool: 是否成功
    """
    svo_name = Path(svo_path).stem
    svo_output_dir = os.path.join(output_dir, svo_name)
    os.makedirs(svo_output_dir, exist_ok=True)
    
    mesh_path = os.path.join(svo_output_dir, f"{svo_name}_mesh.obj")
    
    print("\n" + "="*70)
    print(f"Generating Mesh for: {svo_name}")
    print("="*70)
    print(f"SVO: {svo_path}")
    print(f"Output: {mesh_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查是否已存在
    if os.path.exists(mesh_path):
        print(f"[Info] Mesh already exists: {mesh_path}")
        print("[Info] Skipping...")
        return True
    
    # 初始化ZED
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    init.set_from_svo_file(svo_path)
    
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Cannot open SVO: {status}")
        return False
    
    # 获取SVO信息
    total_frames = zed.get_svo_number_of_frames()
    print(f"[Info] Total frames: {total_frames}")
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Enable Positional Tracking: {returned_state}")
        zed.close()
        return False
    
    # 配置空间映射 - 生成mesh
    print("[Info] Configuring spatial mapping for mesh generation...")
    spatial_mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.HIGH,
        mapping_range=sl.MAPPING_RANGE.SHORT,
        max_memory_usage=8192,
        save_texture=True,
        use_chunk_only=False,
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.MESH  # 生成mesh而不是点云
    )
    
    spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
    spatial_mapping_parameters.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.SHORT)
    spatial_mapping_parameters.save_texture = True
    spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.MESH
    
    zed.enable_spatial_mapping(spatial_mapping_parameters)
    
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    # 处理所有帧
    print("[Info] Processing frames...")
    frame_count = 0
    
    while True:
        grab_result = zed.grab(runtime_parameters)
        
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed.")
            break
        
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[Progress] {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    print(f"[Info] Processed {frame_count} frames")
    
    # 提取mesh
    print("[Info] Extracting mesh...")
    mesh = sl.Mesh()
    extract_status = zed.extract_whole_spatial_map(mesh)
    
    if extract_status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Failed to extract mesh: {extract_status}")
        zed.close()
        return False
    
    print(f"[Info] Mesh extracted successfully!")
    print(f"[Info] Vertices: {len(mesh.vertices)}")
    print(f"[Info] Triangles: {len(mesh.triangles)}")
    
    # 保存mesh
    print(f"[Info] Saving mesh to: {mesh_path}")
    mesh.save(mesh_path, sl.MESH_FILE_FORMAT.OBJ)
    
    # 清理
    mesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    print(f"[Info] Mesh generation completed!")
    print(f"[Info] Output: {mesh_path}")
    
    return True


def get_svo_files(svo_dir: str) -> list:
    """获取目录下所有SVO文件"""
    svo_files = []
    if os.path.exists(svo_dir):
        for file in os.listdir(svo_dir):
            if file.endswith('.svo') or file.endswith('.svo2'):
                svo_files.append(os.path.join(svo_dir, file))
    return sorted(svo_files)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Batch generate mesh files from SVO using SpatialMapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_generate_mesh.py
  python batch_generate_mesh.py --svo-dir "C:/Users/ZYF/Documents/ZED"
  python batch_generate_mesh.py --output-dir "data/damage_detection"
  python batch_generate_mesh.py --skip-existing
        """
    )
    
    parser.add_argument(
        '--svo-dir',
        type=str,
        default=DEFAULT_SVO_DIR,
        help=f'SVO files directory (default: {DEFAULT_SVO_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip SVO files that already have mesh generated'
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.svo_dir):
        print(f"[Error] SVO directory not found: {args.svo_dir}")
        return
    
    # 获取所有SVO文件
    svo_files = get_svo_files(args.svo_dir)
    
    if not svo_files:
        print(f"[Error] No SVO files found in: {args.svo_dir}")
        return
    
    print("="*70)
    print("Batch Mesh Generation")
    print("="*70)
    print(f"SVO Directory: {args.svo_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total SVO files found: {len(svo_files)}")
    print("="*70)
    
    # 打印文件列表
    print("\nSVO files to process:")
    for i, svo_file in enumerate(svo_files, 1):
        file_size = os.path.getsize(svo_file) / (1024*1024)  # MB
        print(f"  {i}. {Path(svo_file).name} ({file_size:.2f} MB)")
    
    # 如果需要跳过已处理的文件
    if args.skip_existing:
        print("\n[Info] Checking for existing meshes...")
        svo_files_to_process = []
        for svo_file in svo_files:
            svo_name = Path(svo_file).stem
            mesh_path = os.path.join(args.output_dir, svo_name, f"{svo_name}_mesh.obj")
            if os.path.exists(mesh_path):
                print(f"  - Skipping {svo_name} (mesh already exists)")
            else:
                svo_files_to_process.append(svo_file)
        svo_files = svo_files_to_process
        print(f"[Info] {len(svo_files)} files to process")
    
    if not svo_files:
        print("\n[Info] All files have been processed. Nothing to do.")
        return
    
    # 批量处理
    print("\n" + "="*70)
    print("Starting batch mesh generation...")
    print("="*70)
    
    start_time = time.time()
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for i, svo_file in enumerate(svo_files, 1):
        print(f"\n\n[{i}/{len(svo_files)}] Processing file...")
        
        file_start = time.time()
        success = generate_mesh_from_svo(svo_file, args.output_dir)
        file_duration = time.time() - file_start
        
        if success:
            success_count += 1
            print(f"⏱️  Duration: {file_duration/60:.1f} minutes")
        else:
            failed_count += 1
            failed_files.append(Path(svo_file).name)
        
        # 显示进度
        progress = (i / len(svo_files)) * 100
        print(f"\n📊 Overall Progress: {i}/{len(svo_files)} ({progress:.1f}%)")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {failed_count}")
    
    # 总结
    total_duration = time.time() - start_time
    
    print("\n" + "="*70)
    print("Batch Mesh Generation Complete!")
    print("="*70)
    print(f"Total files: {len(svo_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
