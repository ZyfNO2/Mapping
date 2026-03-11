# -*- coding: utf-8 -*-
"""
SVO文件点云处理主脚本
支持命令行参数指定SVO文件，或使用默认路径

使用方法:
    python process_svo_main.py                                    # 使用默认SVO文件
    python process_svo_main.py --svo path/to/your.svo2            # 指定SVO文件
    python process_svo_main.py --svo path/to/your.svo2 --output custom_output  # 指定输出目录

处理流程:
1. spatial_mapping_offline - 生成点云
2. point_cloud_processing - 处理点云
3. point_cloud_converter - 转换格式
"""
import sys
import time
import argparse
import pyzed.sl as sl
import os
import open3d as o3d
import pandas as pd
import numpy as np
from pathlib import Path


# 默认SVO文件路径
DEFAULT_SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-17-12.svo2"


def get_output_dir_from_svo(svo_path):
    """从SVO路径生成输出目录名"""
    svo_name = Path(svo_path).stem  # 获取文件名（不含扩展名）
    return f"data/{svo_name}"


def step1_spatial_mapping(svo_path, output_dir):
    """Step 1: 空间映射生成点云"""
    print("="*70)
    print("Step 1: Spatial Mapping - Generating Point Cloud")
    print("="*70)
    
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    
    # SVO文件路径
    init.set_from_svo_file(svo_path)
    print(f"[Sample] Using SVO File input: {svo_path}")

    # 打开相机
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open : {repr(status)}. Exit program.")
        return False
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"Enable Positional Tracking : {repr(status)}. Exit program.")
        return False

    # 配置空间映射参数
    print("[Sample] Using HIGH quality point cloud settings with SHORT range")
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

    # 主循环
    print("Processing SVO file...")
    while True:
        grab_result = zed.grab(runtime_parameters)
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed. Extracting point cloud...")
            if mapping_activated:
                print("Saving FUSED_POINT_CLOUD mapping result...")
                point_cloud = sl.FusedPointCloud()
                extract_status = zed.extract_whole_spatial_map(point_cloud)
                if extract_status != sl.ERROR_CODE.SUCCESS:
                    print(f"Failed to extract point cloud: {extract_status}")
                    zed.close()
                    return False
                else:
                    print("Successfully extracted point cloud")
                    
                    # 保存到输出文件夹
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # ZED SDK默认保存为OBJ格式
                    point_cloud_filepath = os.path.join(output_dir, "point_cloud_gen_high_quality")
                    point_cloud_filepath = point_cloud_filepath.replace('\\', '/')
                    print(f"Saving point cloud to: {point_cloud_filepath}")
                    
                    # 保存为OBJ格式
                    save_status = point_cloud.save(point_cloud_filepath + ".obj")
                    if save_status:
                        print(f"Point cloud saved successfully (OBJ format)")
                        try:
                            print(f"File size: {os.path.getsize(point_cloud_filepath + '.obj') / (1024*1024):.2f} MB")
                        except:
                            pass
                        
                        # 转换为PLY格式供后续处理
                        print("Converting OBJ to PLY format...")
                        
                        vertices = []
                        colors = []
                        with open(point_cloud_filepath + ".obj", 'r') as f:
                            for line in f:
                                if line.startswith('v '):
                                    parts = line.strip().split()
                                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                    r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                                    vertices.append([x, y, z])
                                    colors.append([r, g, b])
                        
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
                        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
                        
                        ply_path = point_cloud_filepath + ".ply"
                        o3d.io.write_point_cloud(ply_path, pcd)
                        print(f"Converted to PLY: {ply_path}")
                        print(f"PLY file size: {os.path.getsize(ply_path) / (1024*1024):.2f} MB")
                    else:
                        print(f"Failed to save point cloud")
                        zed.close()
                        return False
            break
        elif grab_result != sl.ERROR_CODE.SUCCESS:
            print(f"Error during grab: {grab_result}")
            zed.close()
            return False

    # 关闭相机
    zed.close()
    print("Step 1 complete!")
    return True


def step2_point_cloud_processing(output_dir):
    """Step 2: 点云处理"""
    print("\n" + "="*70)
    print("Step 2: Point Cloud Processing")
    print("="*70)
    
    # 使用PLY格式（Step 1已转换）
    input_path = os.path.join(output_dir, "point_cloud_gen_high_quality.ply")
    output_path = os.path.join(output_dir, "point_cloud_processed.ply")
    
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    # 统计异常值移除
    print("Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd.points)} points")
    
    # 体素下采样
    print("Voxel downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print(f"After downsampling: {len(pcd.points)} points")
    
    # 估计法线
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # 保存
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved processed point cloud to: {output_path}")
    print("Step 2 complete!")
    return True


def step3_point_cloud_converter(output_dir):
    """Step 3: 点云转换"""
    print("\n" + "="*70)
    print("Step 3: Point Cloud Converter")
    print("="*70)
    
    input_ply = os.path.join(output_dir, "point_cloud_processed.ply")
    output_csv = os.path.join(output_dir, "point_cloud.csv")
    output_obj = os.path.join(output_dir, "point_cloud.obj")
    
    print("Loading processed point cloud...")
    pcd = o3d.io.read_point_cloud(input_ply)
    print(f"Loaded {len(pcd.points)} points")
    
    # 转换为CSV
    print("Converting to CSV...")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
    
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'r': colors[:, 0],
        'g': colors[:, 1],
        'b': colors[:, 2],
        'nx': normals[:, 0],
        'ny': normals[:, 1],
        'nz': normals[:, 2]
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")
    
    # 转换为OBJ
    print("Converting to OBJ...")
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
        o3d.io.write_triangle_mesh(output_obj, mesh)
        print(f"Saved OBJ to: {output_obj}")
    except Exception as e:
        print(f"OBJ conversion failed: {e}")
        print("Saving as point cloud OBJ instead...")
        o3d.io.write_point_cloud(output_obj.replace('.obj', '_points.obj'), pcd)
    
    print("Step 3 complete!")
    return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Process SVO file to generate point cloud',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python process_svo_main.py
    # Use default SVO file
    
  python process_svo_main.py --svo "path\to\your.svo2"
    # Process specified SVO file
    
  python process_svo_main.py --svo "your.svo2" --output "custom_folder"
    # Specify custom output directory
        """
    )
    
    parser.add_argument(
        '--svo',
        type=str,
        default=DEFAULT_SVO_PATH,
        help=f'Path to SVO file (default: {DEFAULT_SVO_PATH})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data/<svo_filename>)'
    )
    
    args = parser.parse_args()
    
    # 获取SVO路径
    svo_path = args.svo
    
    # 检查SVO文件是否存在
    if not Path(svo_path).exists():
        print(f"Error: SVO file not found: {svo_path}")
        print("\nSearching for available SVO files...")
        
        # 搜索可用的SVO文件
        zed_dir = Path(r"C:\Users\ZYF\Documents\ZED")
        if zed_dir.exists():
            svo_files = list(zed_dir.glob("*.svo2"))
            if svo_files:
                print("\nAvailable SVO files:")
                for i, svo in enumerate(svo_files, 1):
                    print(f"  {i}. {svo.name}")
                print("\nPlease specify one of the above files using --svo")
        return
    
    # 获取输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = get_output_dir_from_svo(svo_path)
    
    print("="*70)
    print("SVO Point Cloud Processing")
    print("="*70)
    print(f"SVO file: {svo_path}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行三个步骤
    success = True
    
    if not step1_spatial_mapping(svo_path, output_dir):
        print("Step 1 failed!")
        success = False
    
    if success and not step2_point_cloud_processing(output_dir):
        print("Step 2 failed!")
        success = False
    
    if success and not step3_point_cloud_converter(output_dir):
        print("Step 3 failed!")
        success = False
    
    # 显示结果
    print("\n" + "="*70)
    if success:
        print("All steps completed successfully!")
        print(f"\nOutput files in: {output_dir}")
        output_path = Path(output_dir)
        for f in sorted(output_path.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  📄 {f.name:<50} {size_mb:>8.2f} MB")
    else:
        print("Some steps failed. Check the output above.")
    print("="*70)


if __name__ == "__main__":
    main()
