"""
文件目的：离线从SVO文件生成高质量点云并保存为PLY格式

处理流程：
1. 初始化ZED相机参数，使用SVO文件作为输入
2. 配置位置跟踪和空间映射参数（使用高质量设置）
3. 启动空间映射过程
4. 处理SVO文件中的每一帧
5. 当SVO文件播放完成时，提取点云数据
6. 保存点云数据为PLY格式

数据处理方式：
- 从SVO文件中读取深度信息
- 使用ZED SDK的空间映射功能生成高质量点云
- 将点云数据保存为PLY格式（包含颜色信息）
"""
import sys
import time
import pyzed.sl as sl
import os


def main():
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # 使用神经网络深度模式，获得最佳深度质量
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL的坐标系
    init.depth_maximum_distance = 10.  # 增加最大深度距离
    
    # 硬编码SVO文件路径
    svo_path = "C:\\Users\\ZYF\\Documents\\ZED\\HD2K_SN36245620_15-16-49.svo2"
    init.set_from_svo_file(svo_path)
    print("[Sample] Using SVO File input: {0}".format(svo_path))

    # 打开相机
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    # 配置位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        exit()

    # 配置空间映射参数（使用最高质量设置，近距离采集）
    print("[Sample] Using HIGH quality point cloud settings with SHORT range")
    spatial_mapping_parameters = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.HIGH,  # 最高分辨率
        mapping_range=sl.MAPPING_RANGE.SHORT,  # 短距离映射
        max_memory_usage=8192,  # 增加内存使用以获得更高质量
        save_texture=True,  # 保存纹理信息
        use_chunk_only=False,  # 不使用仅块模式，获得完整点云
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
    )
    
    # 进一步优化参数
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
        # Grab an image
        grab_result = zed.grab(runtime_parameters)
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed. Extracting point cloud...")
            # 自动结束映射
            if mapping_activated:
                # 保存FUSED_POINT_CLOUD mapping result...
                print("Saving FUSED_POINT_CLOUD mapping result...")
                # 提取点云 - 直接从当前映射中提取
                point_cloud = sl.FusedPointCloud()
                extract_status = zed.extract_whole_spatial_map(point_cloud)
                if extract_status != sl.ERROR_CODE.SUCCESS:
                    print(f"Failed to extract point cloud: {extract_status}")
                else:
                    print("Successfully extracted point cloud")
                    
                    # 确保data目录存在
                    os.makedirs('data', exist_ok=True)
                    
                    # 保存点云为PLY格式
                    point_cloud_ply_filepath = "data/point_cloud_gen_high_quality.ply"
                    # 使用PLY格式保存
                    status = point_cloud.save(point_cloud_ply_filepath, sl.MESH_FILE_FORMAT.PLY)
                    if status:
                        print("Point cloud saved as PLY: " + point_cloud_ply_filepath)
                        # 检查文件是否存在且大小大于0
                        if os.path.exists(point_cloud_ply_filepath):
                            print(f"Point cloud file size: {os.path.getsize(point_cloud_ply_filepath)} bytes")
                        else:
                            print(f"Warning: Point cloud file {point_cloud_ply_filepath} does not exist")
                    else:
                        print("Failed to save the point cloud as PLY")
                
                point_cloud.clear()
                mapping_activated = False
            break
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            # 更新位姿数据
            zed.get_position(pose)

    # 清理资源
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    
    # Free allocated memory before closing the camera
    image.free(memory_type=sl.MEM.CPU)
    
    # Close the ZED
    zed.close()
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()