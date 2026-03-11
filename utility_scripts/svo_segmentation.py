"""
SVO文件裂缝分割处理
读取SVO文件，使用YOLO模型进行裂缝分割，生成带mask的逐帧图和MP4视频

处理流程：
1. 读取SVO文件
2. 加载YOLO分割模型 (best.pt)
3. 对每一帧进行裂缝分割
4. 生成带mask的逐帧图
5. 合成带mask的MP4视频
"""
import sys
import time
import cv2
import numpy as np
import pyzed.sl as sl
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def apply_mask_overlay(image, mask, alpha=0.5, color=(0, 0, 255)):
    """
    将分割mask叠加到图像上
    
    Args:
        image: 原始图像 (H, W, 3)
        mask: 分割mask (H, W) 或 (H, W, 1)
        alpha: mask透明度
        color: mask颜色 (B, G, R)
    
    Returns:
        叠加后的图像
    """
    overlay = image.copy()
    
    # 确保mask是2D
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    # 将mask转换为bool类型
    mask_bool = mask > 0.5
    
    # 创建彩色mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask_bool] = color
    
    # 叠加
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha,
        colored_mask[mask_bool], alpha, 0
    )
    
    return overlay


def process_svo_segmentation(
    svo_path,
    model_path,
    output_dir="output_segmentation",
    save_frames=True,
    save_video=True,
    fps=30
):
    """
    处理SVO文件进行裂缝分割
    
    Args:
        svo_path: SVO文件路径
        model_path: YOLO模型路径 (.pt文件)
        output_dir: 输出目录
        save_frames: 是否保存逐帧图
        save_video: 是否保存MP4视频
        fps: 输出视频帧率
    """
    print("="*70)
    print("SVO Crack Segmentation Processing")
    print("="*70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载YOLO模型
    print(f"\nLoading YOLO model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # 初始化ZED
    print(f"\nOpening SVO file: {svo_path}")
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {status}")
        return
    
    # 获取SVO信息 (兼容不同SDK版本)
    try:
        svo_info = zed.get_svo_info()
        total_frames = svo_info.get_number_of_frames()
    except AttributeError:
        # 旧版本SDK可能没有get_svo_info方法
        # 通过遍历获取总帧数
        print("Counting frames...")
        total_frames = 0
        zed.set_svo_position(0)
        while zed.grab() == sl.ERROR_CODE.SUCCESS:
            total_frames += 1
        zed.set_svo_position(0)
    
    print(f"Total frames: {total_frames}")
    
    # 准备视频写入
    video_writer = None
    if save_video:
        # 获取第一帧来确定尺寸
        zed.set_svo_position(0)
        zed.grab()
        image_zed = sl.Mat()
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame = image_zed.get_data()
        height, width = frame.shape[:2]
        
        video_path = output_path / "segmentation_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        print(f"Video output: {video_path}")
        
        # 重置到开始
        zed.set_svo_position(0)
    
    # 处理每一帧
    print("\nProcessing frames...")
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            # 读取帧
            grab_result = zed.grab()
            if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            
            if grab_result != sl.ERROR_CODE.SUCCESS:
                continue
            
            # 获取图像
            image_zed = sl.Mat()
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            
            # 转换为RGB (ZED输出BGRA)
            if frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = frame.copy()
            
            # YOLO分割
            results = model(frame_rgb, verbose=False)
            
            # 获取分割结果
            annotated_frame = frame_bgr.copy()
            
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # 处理每个mask
                for mask in masks:
                    # 调整mask尺寸到图像尺寸
                    mask_resized = cv2.resize(
                        mask, 
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # 叠加mask (红色半透明)
                    annotated_frame = apply_mask_overlay(
                        annotated_frame, 
                        mask_resized, 
                        alpha=0.4, 
                        color=(0, 0, 255)  # BGR红色
                    )
                
                # 添加检测框和标签
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        # 画框
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 添加置信度标签
                        label = f"crack: {conf:.2f}"
                        cv2.putText(
                            annotated_frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2
                        )
            
            # 添加帧号
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2
            )
            
            # 保存逐帧图
            if save_frames:
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
            
            # 写入视频
            if save_video and video_writer is not None:
                video_writer.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # 释放资源
    zed.close()
    if video_writer is not None:
        video_writer.release()
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Total frames processed: {frame_count}")
    if save_frames:
        print(f"Frames saved to: {frames_dir}")
    if save_video:
        print(f"Video saved to: {output_path / 'segmentation_result.mp4'}")
    print("="*70)


def main():
    """主函数"""
    # 配置参数
    SVO_PATH = "C:\\Users\\ZYF\\Documents\\ZED\\HD2K_SN36245620_15-16-49.svo2"
    MODEL_PATH = "runs/segment/ad-astra/heuristic-elephant/train/weights/best.pt"
    OUTPUT_DIR = "output_segmentation"
    
    # 检查文件是否存在
    if not Path(SVO_PATH).exists():
        print(f"Error: SVO file not found: {SVO_PATH}")
        print("Please check the SVO file path.")
        return
    
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Searching for available models...")
        
        # 搜索可用的best.pt
        base_path = Path(".")
        best_models = list(base_path.rglob("best.pt"))
        
        if best_models:
            print("\nAvailable best.pt models:")
            for i, model in enumerate(best_models, 1):
                print(f"{i}. {model}")
            
            # 使用第一个找到的模型
            MODEL_PATH = str(best_models[0])
            print(f"\nUsing: {MODEL_PATH}")
        else:
            print("No best.pt found! Please train a model first.")
            return
    
    # 运行处理
    process_svo_segmentation(
        svo_path=SVO_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        save_frames=True,
        save_video=True,
        fps=30
    )


if __name__ == "__main__":
    main()
