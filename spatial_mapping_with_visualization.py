"""
文件目的：使用ZED相机进行空间映射和点云生成

处理流程：
1. 初始化ZED相机参数，使用SVO文件作为输入
2. 配置位置跟踪和空间映射参数
3. 启动空间映射过程
4. 处理SVO文件中的每一帧
5. 当SVO文件播放完成时，提取点云数据
6. 保存点云数据为CSV和PLY格式

数据处理方式：
- 从SVO文件中读取深度信息
- 使用ZED SDK的空间映射功能生成点云
- 将点云数据保存为CSV格式（包含顶点坐标）
- 将点云数据保存为PLY格式（包含颜色信息）
"""
import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse


def main(high_quality=False):
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # OpenGL's coordinate system is right_handed    
    init.depth_maximum_distance = 8.
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
    
    camera_infos = zed.get_camera_information()
    pose = sl.Pose()

    # 配置位置跟踪
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    # 配置PositionalTrackingParameters
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # 启用区域内存，用于环境记忆和重定位
    positional_tracking_parameters.enable_area_memory = True
    # 启用位姿平滑，减少抖动
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        exit()

    # 配置空间映射参数
    # 根据high_quality参数设置不同的映射参数
    if high_quality:
        print("[Sample] Using HIGH quality point cloud settings")
        # 高分辨率设置
        spatial_mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.HIGH,mapping_range =  sl.MAPPING_RANGE.MEDIUM,max_memory_usage = 4096,save_texture = False,use_chunk_only = False,reverse_vertex_order = False,map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD)
    else:
        print("[Sample] Using LOW quality point cloud settings")
        # 低分辨率设置
        spatial_mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.LOW,mapping_range =  sl.MAPPING_RANGE.MEDIUM,max_memory_usage = 512,save_texture = False,use_chunk_only = True,reverse_vertex_order = False,map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD)
    
    pymesh = sl.FusedPointCloud()

    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    mapping_activated = False

    image = sl.Mat()  
    point_cloud = sl.Mat()
    pose = sl.Pose()

    viewer = gl.GLViewer()

    viewer.init(zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, pymesh, 0)
    # 移除提示信息
    # print("Press on 'Space' to enable / disable spatial mapping")
    # print("Disable the spatial mapping after enabling it will output a .obj mesh file")
    
    # 自动开始映射
    init_pose = sl.Transform()
    zed.reset_positional_tracking(init_pose) 

    # Configure spatial mapping parameters based on quality setting
    if high_quality:
        spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)
        spatial_mapping_parameters.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.MEDIUM)
        spatial_mapping_parameters.use_chunk_only = False
    else:
        spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.LOW)
        spatial_mapping_parameters.range_meter = sl.SpatialMappingParameters().get_range_preset(sl.MAPPING_RANGE.MEDIUM)
        spatial_mapping_parameters.use_chunk_only = True
        # 进一步降低点云密度的参数
        spatial_mapping_parameters.max_memory_usage = 512
    
    spatial_mapping_parameters.save_texture = False         # Set to True to apply texture over the created mesh
    spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

    # Enable spatial mapping with point cloud type
    zed.enable_spatial_mapping(spatial_mapping_parameters)

    # Clear previous mesh data
    pymesh.clear()
    viewer.clear_current_mesh()

    # Start timer
    last_call = time.time()

    mapping_activated = True

    # 主循环
    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        grab_result = zed.grab(runtime_parameters)
        if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO file playback completed. Exiting...")
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
                    
                    # 保存vertices数据为CSV
                    import numpy as np
                    import csv
                    import os
                    # 确保data目录存在
                    os.makedirs('data', exist_ok=True)
                    # 先更新vertices数据
                    point_cloud.update_from_chunklist()
                    vertices = point_cloud.vertices
                    # 提取xyz数据，忽略rgba
                    xyz_data = vertices[:, :3]
                    # 保存为CSV文件
                    csv_filepath = "data/point_cloud_vertices.csv"
                    with open(csv_filepath, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        # 写入表头
                        csv_writer.writerow(['x', 'y', 'z'])
                        # 写入数据
                        csv_writer.writerows(xyz_data)
                    print(f"Vertices data saved as CSV: {csv_filepath}")
                    
                    # 保存点云为PLY格式
                    point_cloud_ply_filepath = "data/point_cloud_gen.ply"
                    # 使用PLY格式保存
                    status = point_cloud.save(point_cloud_ply_filepath, sl.MESH_FILE_FORMAT.PLY)
                    if status:
                        print("Point cloud saved as PLY: " + point_cloud_ply_filepath)
                        # 检查文件是否存在且大小大于0
                        import os
                        if os.path.exists(point_cloud_ply_filepath):
                            print(f"Point cloud file size: {os.path.getsize(point_cloud_ply_filepath)} bytes")
                        else:
                            print(f"Warning: Point cloud file {point_cloud_ply_filepath} does not exist")
                    else:
                        print("Failed to save the point cloud as PLY")
                
                point_cloud.clear()
                
                mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                mapping_activated = False
            break
        if grab_result <= sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.get_position(pose)

            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                # Compute elapsed time since the last call of Camera.request_spatial_map_async()
                duration = time.time() - last_call
                # Ask for a mesh update if 500ms elapsed since last request
                if(duration > .5 and viewer.chunks_updated()):
                    zed.request_spatial_map_async()
                    last_call = time.time()

                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()

            # 调用update_view，但忽略返回值，因为我们不需要手动触发状态变化
            viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

    # 清理资源
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    
    # Free allocated memory before closing the camera
    pymesh.clear()
    image.free(memory_type=sl.MEM.CPU)
    
    # Close the ZED
    zed.close()

if __name__ == "__main__":
    # 命令行参数处理
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_quality', action='store_true', help='Enable high quality point cloud generation with higher resolution')
    args = parser.parse_args()
    main(args.high_quality)
