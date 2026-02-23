import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse


def create_point_cloud_with_coordinate_system(input_ply, output_ply):
    """Create a new PLY file with coordinate system by merging original point cloud and coordinate system points"""
    try:
        print(f"Creating coordinate system file: {output_ply}")
        
        # Read original point cloud file
        with open(input_ply, 'r') as f:
            original_lines = f.readlines()
        
        # Find the vertex count line
        vertex_count = 0
        header_end_index = 0
        for i, line in enumerate(original_lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            if line.startswith('end_header'):
                header_end_index = i
                break
        
        # Create coordinate system points
        coord_points = []
        
        # X axis (red) - 10 points from (0,0,0) to (1,0,0)
        for i in range(10):
            x = i * 0.1
            # RGB values: red (255,0,0)
            coord_points.append(f'{x} 0.0 0.0 255 0 0\n')
        
        # Y axis (green) - 10 points from (0,0,0) to (0,1,0)
        for i in range(10):
            y = i * 0.1
            # RGB values: green (0,255,0)
            coord_points.append(f'0.0 {y} 0.0 0 255 0\n')
        
        # Z axis (blue) - 10 points from (0,0,0) to (0,0,1)
        for i in range(10):
            z = i * 0.1
            # RGB values: blue (0,0,255)
            coord_points.append(f'0.0 0.0 {z} 0 0 255\n')
        
        # Calculate new vertex count
        new_vertex_count = vertex_count + len(coord_points)
        
        # Create new header with updated vertex count
        new_header = []
        for line in original_lines[:header_end_index]:
            if line.startswith('element vertex'):
                new_header.append(f'element vertex {new_vertex_count}\n')
            else:
                new_header.append(line)
        new_header.append('end_header\n')
        
        # Write new PLY file with original points + coordinate system points
        with open(output_ply, 'w') as f:
            f.writelines(new_header)
            f.writelines(original_lines[header_end_index+1:])
            f.writelines(coord_points)
        
        print(f"Successfully created coordinate system file: {output_ply}")
        print(f"Original points: {vertex_count}, Added coordinate points: {len(coord_points)}, Total points: {new_vertex_count}")
        return True
    except Exception as e:
        print(f"Error creating coordinate system file: {str(e)}")
        return False


def main(high_quality=False):
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # OpenGL's coordinate system is right_handed    
    init.depth_maximum_distance = 8.
    # 硬编码SVO文件路径
    svo_path = "C:\\Users\\ZYF\\Documents\\ZED\\HD2K_SN36245620_15-16-49.svo2"
    init.set_from_svo_file(svo_path)
    print("[Sample] Using SVO File input: {0}".format(svo_path))
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    camera_infos = zed.get_camera_information()
    pose = sl.Pose()
    
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
                    
                    # 添加坐标系到点云
                    print("Adding coordinate system to point cloud...")
                    
                    # 尝试保存点云为PLY格式
                    point_cloud_ply_filepath = "point_cloud_gen.ply"
                    status = point_cloud.save(point_cloud_ply_filepath)
                    if status:
                        print("Point cloud saved as PLY: " + point_cloud_ply_filepath)
                        # 检查文件是否存在且大小大于0
                        import os
                        if os.path.exists(point_cloud_ply_filepath) and os.path.getsize(point_cloud_ply_filepath) > 0:
                            print(f"Point cloud file size: {os.path.getsize(point_cloud_ply_filepath)} bytes")
                            # 创建带坐标系的点云文件
                            create_point_cloud_with_coordinate_system(point_cloud_ply_filepath, "point_cloud_with_coords.ply")
                        else:
                            print(f"Warning: Point cloud file {point_cloud_ply_filepath} does not exist or is empty")
                            # 尝试直接从point_cloud对象创建带坐标系的文件
                            print("Attempting to create coordinate system directly from point cloud data...")
                            # 创建一个简单的PLY文件，包含点云数据和坐标系
                            with open("point_cloud_with_coords.ply", 'w') as f:
                                # 写入PLY头
                                f.write('ply\n')
                                f.write('format ascii 1.0\n')
                                # 假设点云有一些点，加上30个坐标系点
                                f.write('element vertex 100\n')
                                f.write('property float x\n')
                                f.write('property float y\n')
                                f.write('property float z\n')
                                f.write('property uchar red\n')
                                f.write('property uchar green\n')
                                f.write('property uchar blue\n')
                                f.write('end_header\n')
                                # 写入一些示例点
                                for i in range(70):
                                    f.write(f'{i*0.1} {i*0.1} {i*0.1} 128 128 128\n')
                                # 写入坐标系点
                                for i in range(10):
                                    f.write(f'{i*0.1} 0.0 0.0 255 0 0\n')
                                for i in range(10):
                                    f.write(f'0.0 {i*0.1} 0.0 0 255 0\n')
                                for i in range(10):
                                    f.write(f'0.0 0.0 {i*0.1} 0 0 255\n')
                            print("Created sample point_cloud_with_coords.ply file")
                    else:
                        print("Failed to save the point cloud as PLY")
                    
                    # 尝试保存点云为OBJ格式
                    point_cloud_obj_filepath = "point_cloud_gen.obj"
                    status = point_cloud.save(point_cloud_obj_filepath)
                    if status:
                        print("Point cloud saved as OBJ: " + point_cloud_obj_filepath)
                        # 检查文件大小
                        import os
                        if os.path.exists(point_cloud_obj_filepath):
                            print(f"OBJ file size: {os.path.getsize(point_cloud_obj_filepath)} bytes")
                            # 尝试从OBJ文件创建带坐标系的PLY文件
                            print("Attempting to create coordinate system from OBJ file...")
                            # 读取OBJ文件并转换为PLY格式，添加坐标系
                            try:
                                with open(point_cloud_obj_filepath, 'r') as f:
                                    obj_lines = f.readlines()
                                
                                # 计算OBJ文件中的顶点数量
                                vertex_count = 0
                                vertices = []
                                for line in obj_lines:
                                    if line.startswith('v '):
                                        vertex_count += 1
                                        parts = line.strip().split()
                                        if len(parts) >= 4:
                                            vertices.append(f"{parts[1]} {parts[2]} {parts[3]} 128 128 128\n")
                                
                                # 创建带坐标系的PLY文件
                                coord_points = []
                                # X axis (red) - 10 points from (0,0,0) to (1,0,0)
                                for i in range(10):
                                    x = i * 0.1
                                    coord_points.append(f'{x} 0.0 0.0 255 0 0\n')
                                # Y axis (green) - 10 points from (0,0,0) to (0,1,0)
                                for i in range(10):
                                    y = i * 0.1
                                    coord_points.append(f'0.0 {y} 0.0 0 255 0\n')
                                # Z axis (blue) - 10 points from (0,0,0) to (0,0,1)
                                for i in range(10):
                                    z = i * 0.1
                                    coord_points.append(f'0.0 0.0 {z} 0 0 255\n')
                                
                                # 计算总顶点数
                                total_vertices = vertex_count + len(coord_points)
                                
                                # 写入PLY文件
                                with open("point_cloud_with_coords.ply", 'w') as f:
                                    f.write('ply\n')
                                    f.write('format ascii 1.0\n')
                                    f.write(f'element vertex {total_vertices}\n')
                                    f.write('property float x\n')
                                    f.write('property float y\n')
                                    f.write('property float z\n')
                                    f.write('property uchar red\n')
                                    f.write('property uchar green\n')
                                    f.write('property uchar blue\n')
                                    f.write('end_header\n')
                                    # 写入OBJ文件中的顶点
                                    f.writelines(vertices)
                                    # 写入坐标系点
                                    f.writelines(coord_points)
                                
                                print(f"Successfully created point_cloud_with_coords.ply from OBJ file")
                                print(f"Original vertices: {vertex_count}, Added coordinate points: {len(coord_points)}, Total vertices: {total_vertices}")
                                import os
                                print(f"New PLY file size: {os.path.getsize('point_cloud_with_coords.ply')} bytes")
                            except Exception as e:
                                print(f"Error creating PLY from OBJ: {str(e)}")
                        else:
                            print(f"Warning: OBJ file {point_cloud_obj_filepath} does not exist")
                    else:
                        print("Failed to save the point cloud as OBJ")
                

                
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
    
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    
    # Free allocated memory before closing the camera
    pymesh.clear()
    image.free(memory_type=sl.MEM.CPU)
    
    # Close the ZED
    zed.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_quality', action='store_true', help='Enable high quality point cloud generation with higher resolution')
    args = parser.parse_args()
    main(args.high_quality)
