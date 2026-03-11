"""
从SVO文件中提取第一帧图像，并计算图像中心点在3D空间中的坐标
功能：
1. 读取SVO文件的第一帧（自适应分辨率）
2. 保存第一帧彩色图像（带采样点标注）
3. 计算图像中心点对应的3D坐标
4. 输出坐标到文本文件
5. 预览点云并在坐标处显示洋红色球
"""
import pyzed.sl as sl
import os
import cv2
import numpy as np
import open3d as o3d


def get_first_frame_center_3d(svo_path: str, output_dir: str = "data") -> tuple:
    """
    从SVO文件中提取第一帧，并计算中心点的3D坐标
    
    Args:
        svo_path: SVO文件路径
        output_dir: 输出目录
        
    Returns:
        tuple: (success: bool, coordinate: tuple(x, y, z) or None, image_path: str)
    """
    # 初始化参数设置
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = 10.
    
    init.set_from_svo_file(svo_path)
    print(f"[Info] 使用SVO文件: {svo_path}")

    # 打开相机
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"[Error] 相机打开失败: {status}")
        return False, None, None
    
    # 获取相机信息（分辨率等）
    camera_info = zed.get_camera_information()
    image_size = camera_info.camera_configuration.resolution
    print(f"[Info] 视频分辨率: {image_size.width}x{image_size.height}")
    
    # 配置位置跟踪（获取3D坐标需要）
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] 位置跟踪启用失败: {returned_state}")
        zed.close()
        return False, None, None

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    # 抓取第一帧
    grab_result = zed.grab(runtime_parameters)
    if grab_result != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] 抓取第一帧失败: {grab_result}")
        zed.disable_positional_tracking()
        zed.close()
        return False, None, None
    
    # 获取图像和深度数据
    image = sl.Mat()
    depth_map = sl.Mat()
    point_cloud_frame = sl.Mat()
    
    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
    zed.retrieve_measure(point_cloud_frame, sl.MEASURE.XYZ)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像尺寸和中心点
    image_width = image.get_width()
    image_height = image.get_height()
    center_x = image_width // 2
    center_y = image_height // 2
    print(f"[Info] 图像尺寸: {image_width}x{image_height}, 中心点: ({center_x}, {center_y})")
    
    # 获取中心点的深度值
    depth_value = depth_map.get_value(center_x, center_y)[1]
    print(f"[Info] 中心点深度值: {depth_value} 米")
    
    center_3d = None
    
    if depth_value > 0 and depth_value != float('inf'):
        # 获取中心点的3D坐标
        point_3d = point_cloud_frame.get_value(center_x, center_y)[1]
        center_3d = (point_3d[0], point_3d[1], point_3d[2])
        print(f"[Info] 中心点在点云中的3D坐标: X={point_3d[0]:.4f}, Y={point_3d[1]:.4f}, Z={point_3d[2]:.4f}")
        
        # 保存坐标到文件
        center_coord_path = os.path.join(output_dir, "first_frame_center_coordinate.txt")
        with open(center_coord_path, 'w', encoding='utf-8') as f:
            f.write("第一帧图片中心点在点云中的坐标:\n")
            f.write(f"SVO文件: {svo_path}\n")
            f.write(f"图像尺寸: {image_width}x{image_height}\n")
            f.write(f"中心像素坐标: ({center_x}, {center_y})\n")
            f.write(f"深度值: {depth_value:.4f} 米\n")
            f.write(f"3D坐标 (X, Y, Z): ({point_3d[0]:.6f}, {point_3d[1]:.6f}, {point_3d[2]:.6f})\n")
            f.write(f"坐标单位: 米 (RIGHT_HANDED_Y_UP坐标系)\n")
        print(f"[Info] 中心点坐标已保存至: {center_coord_path}")
    else:
        print(f"[Warning] 中心点深度值无效 ({depth_value})，无法计算3D坐标")
    
    # 保存带标注的图片
    first_frame_path = save_annotated_image(image, center_x, center_y, output_dir)
    
    # 清理资源
    image.free(memory_type=sl.MEM.CPU)
    depth_map.free(memory_type=sl.MEM.CPU)
    point_cloud_frame.free(memory_type=sl.MEM.CPU)
    zed.disable_positional_tracking()
    zed.close()
    
    return center_3d is not None, center_3d, first_frame_path


def save_annotated_image(image: sl.Mat, center_x: int, center_y: int, output_dir: str) -> str:
    """
    保存带采样点标注的图片
    
    Args:
        image: ZED图像
        center_x: 中心点X坐标
        center_y: 中心点Y坐标
        output_dir: 输出目录
        
    Returns:
        str: 保存的图片路径
    """
    # 将ZED图像转换为OpenCV格式
    image_cv = image.get_data()
    
    # 根据图像大小调整标记大小
    image_height, image_width = image_cv.shape[:2]
    circle_radius = max(5, min(image_width, image_height) // 100)
    line_length = circle_radius * 3
    thickness = max(2, circle_radius // 3)
    
    # 绘制十字准星
    color = (0, 255, 255)  # 黄色 (BGR格式)
    
    # 水平线
    cv2.line(image_cv, 
             (center_x - line_length, center_y), 
             (center_x + line_length, center_y), 
             color, thickness)
    
    # 垂直线
    cv2.line(image_cv, 
             (center_x, center_y - line_length), 
             (center_x, center_y + line_length), 
             color, thickness)
    
    # 绘制中心圆
    cv2.circle(image_cv, (center_x, center_y), circle_radius, color, thickness)
    
    # 绘制内圆点
    cv2.circle(image_cv, (center_x, center_y), circle_radius // 3, color, -1)
    
    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(image_width, image_height) / 2000)
    text = f"Sample Point: ({center_x}, {center_y})"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # 文字背景
    padding = 10
    cv2.rectangle(image_cv, 
                  (center_x + circle_radius + 10, center_y - text_size[1] - padding),
                  (center_x + circle_radius + 10 + text_size[0] + padding * 2, center_y + padding),
                  (0, 0, 0), -1)
    
    # 文字
    cv2.putText(image_cv, text, 
                (center_x + circle_radius + 10 + padding, center_y),
                font, font_scale, color, thickness)
    
    # 保存图片
    first_frame_path = os.path.join(output_dir, "first_frame_annotated.png")
    cv2.imwrite(first_frame_path, image_cv)
    print(f"[Info] 带标注的第一帧图片已保存至: {first_frame_path}")
    
    return first_frame_path


def visualize_point_cloud_with_sphere(point_cloud_path: str, sphere_center: tuple, 
                                       sphere_radius: float = 0.007,  # 7mm = 0.007m
                                       output_dir: str = "data"):
    """
    预览点云并在指定坐标处显示洋红色球
    
    Args:
        point_cloud_path: 点云文件路径
        sphere_center: 球心坐标 (x, y, z)
        sphere_radius: 球半径（米），默认7mm
        output_dir: 输出目录
    """
    print(f"[Info] 加载点云文件: {point_cloud_path}")
    
    # 读取点云
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    if len(pcd.points) == 0:
        print("[Error] 点云文件为空")
        return
    
    print(f"[Info] 点云包含 {len(pcd.points)} 个点")
    
    # 创建洋红色球
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=20)
    
    # 洋红色 (Magenta): RGB(255, 0, 255)
    magenta_color = [1.0, 0.0, 1.0]  # Open3D使用0-1范围
    sphere.paint_uniform_color(magenta_color)
    
    # 移动球到指定位置
    sphere.translate(sphere_center)
    
    print(f"[Info] 在坐标 {sphere_center} 处创建洋红色球，半径 {sphere_radius*1000:.1f}mm")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud with Sample Point", width=1280, height=720)
    
    # 添加点云和球
    vis.add_geometry(pcd)
    vis.add_geometry(sphere)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = [0.1, 0.1, 0.1]  # 深灰色背景
    
    # 设置相机视角
    ctr = vis.get_view_control()
    # 将视角对准球的位置
    ctr.set_lookat(sphere_center)
    
    print("[Info] 按 'Q' 或关闭窗口退出预览")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()
    
    # 保存带球的点云
    output_path = os.path.join(output_dir, "point_cloud_with_sphere.ply")
    
    # 将球转换为点云并合并
    sphere_pcd = sphere.sample_points_uniformly(number_of_points=1000)
    sphere_pcd.paint_uniform_color(magenta_color)
    
    combined_pcd = pcd + sphere_pcd
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"[Info] 带标记球的点云已保存至: {output_path}")


def main():
    # SVO文件路径
    svo_path = "C:\\Users\\ZYF\\Documents\\ZED\\HD2K_SN36245620_15-16-49.svo2"
    
    # 点云文件路径（用于预览）
    point_cloud_path = "data/point_cloud_gen_high_quality.ply"
    
    print("=" * 60)
    print("第一帧中心点3D坐标提取工具")
    print("=" * 60)
    
    # 提取第一帧和中心点坐标
    success, coordinate, image_path = get_first_frame_center_3d(svo_path)
    
    print("=" * 60)
    if success:
        print(f"处理成功！中心点3D坐标: {coordinate}")
        print(f"标注图片: {image_path}")
        
        # 如果点云文件存在，进行可视化
        if os.path.exists(point_cloud_path):
            print("\n[Info] 正在启动点云预览...")
            visualize_point_cloud_with_sphere(point_cloud_path, coordinate)
        else:
            print(f"\n[Warning] 点云文件不存在: {point_cloud_path}")
            print("[Info] 跳过点云预览")
    else:
        print("处理失败！")
    print("=" * 60)


if __name__ == "__main__":
    main()
