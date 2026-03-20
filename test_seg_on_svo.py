# -*- coding: utf-8 -*-
"""
SVO分割模型测试脚本
使用YOLO分割模型对SVO文件进行逐帧损伤检测
生成：1) 逐帧检测MP4视频  2) 检测到损伤的序列帧图片

Python环境: G:\Anaconda\envs\zed\python.exe
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import pyzed.sl as sl

# 尝试导入YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("[错误] 无法导入YOLO，请确保ultralytics已安装")
    print("运行: pip install ultralytics")
    sys.exit(1)

# ==================== 配置区域 ====================
# 模型路径
MODEL_PATH = r"G:\Zed\spatial mapping\python\seg.pt"  # 分割模型路径

# SVO文件路径 (与svo_damage_detection.py使用相同的SVO)
SVO_PATH = r"C:\Users\ZYF\Documents\ZED\HD2K_SN36245620_15-17-12.svo2"  # SVO2格式文件

# 输出目录
OUTPUT_DIR = r"G:\Zed\spatial mapping\python\test_output"

# 检测参数
CONF_THRESHOLD = 0.25  # 置信度阈值
SAVE_EVERY_N_FRAMES = 1  # 每N帧保存一张损伤帧 (1=每帧都保存)

# 可视化参数
MASK_ALPHA = 0.5  # 掩码透明度
MASK_COLOR = (0, 0, 255)  # 红色掩码 (BGR格式)
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
# =================================================


class SVOSegmentationTester:
    """SVO分割模型测试器"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        初始化测试器
        
        Args:
            model_path: 分割模型路径
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        
        # 加载模型
        print(f"[INFO] 加载分割模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"[INFO] 模型加载完成")
        
        # 获取模型信息
        print(f"[INFO] 模型任务类型: {self.model.task}")
        print(f"[INFO] 模型类别: {self.model.names}")
    
    def detect(self, image: np.ndarray) -> tuple:
        """
        对单帧图像进行分割检测
        
        Args:
            image: BGR格式图像
            
        Returns:
            (masks, confidences, boxes, annotated_frame)
        """
        # 转换RGB用于YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 运行分割检测
        results = self.model(image_rgb, verbose=False, conf=self.conf_threshold)
        
        masks = []
        confidences = []
        boxes = []
        
        if len(results) > 0 and results[0].masks is not None:
            # 获取分割掩码
            for i, mask in enumerate(results[0].masks.data):
                conf = results[0].boxes.conf[i].item()
                box = results[0].boxes.xyxy[i].cpu().numpy()
                
                masks.append(mask.cpu().numpy())
                confidences.append(conf)
                boxes.append(box)
        
        # 获取带标注的帧
        annotated_frame = results[0].plot() if len(results) > 0 else image_rgb
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        return masks, confidences, boxes, annotated_frame
    
    def process_svo(self, svo_path: str, output_dir: str):
        """
        处理SVO文件
        
        Args:
            svo_path: SVO文件路径
            output_dir: 输出目录
        """
        if not os.path.exists(svo_path):
            raise FileNotFoundError(f"SVO文件不存在: {svo_path}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(output_dir) / f"seg_test_{timestamp}"
        frames_dir = output_base / "damage_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[INFO] 开始处理SVO文件")
        print(f"[INFO] SVO路径: {svo_path}")
        print(f"[INFO] 输出目录: {output_base}")
        print(f"{'='*60}\n")
        
        # 初始化ZED相机
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"无法打开SVO文件: {status}")
        
        # 获取SVO信息
        fps = 30  # 默认FPS
        total_frames = zed.get_svo_number_of_frames()
        print(f"[INFO] SVO总帧数: {total_frames}")
        print(f"[INFO] 视频FPS: {fps}")
        
        # 准备视频写入器
        video_path = output_base / "detection_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 先读取一帧获取分辨率
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        
        frame_count = 0
        damage_frame_count = 0
        video_writer = None
        
        print(f"[INFO] 开始逐帧处理...")
        
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            
            # 获取图像
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()
            
            # 转换为BGR (ZED输出是BGRA)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            frame_count += 1
            
            # 初始化视频写入器
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                print(f"[INFO] 视频分辨率: {w}x{h}")
            
            # 运行分割检测
            masks, confidences, boxes, annotated_frame = self.detect(frame)
            
            # 添加帧信息
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(masks)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 写入视频
            video_writer.write(annotated_frame)
            
            # 如果检测到损伤，保存帧
            if len(masks) > 0 and frame_count % SAVE_EVERY_N_FRAMES == 0:
                damage_frame_count += 1
                frame_filename = frames_dir / f"damage_frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), annotated_frame)
                
                # 同时保存原始帧用于对比
                orig_filename = frames_dir / f"original_frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(orig_filename), frame)
                
                if damage_frame_count <= 5:  # 只显示前5次的日志
                    print(f"[检测到损伤] 帧 {frame_count}, 置信度: {[f'{c:.2f}' for c in confidences]}")
                elif damage_frame_count == 6:
                    print("[INFO] ... (后续检测日志省略)")
            
            # 显示进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[进度] {frame_count}/{total_frames} 帧 ({progress:.1f}%) - 发现 {damage_frame_count} 帧损伤")
        
        # 清理
        zed.close()
        if video_writer:
            video_writer.release()
        
        # 输出统计
        print(f"\n{'='*60}")
        print(f"[完成] SVO处理完成!")
        print(f"[统计] 总帧数: {frame_count}")
        print(f"[统计] 损伤帧数: {damage_frame_count}")
        print(f"[输出] 视频: {video_path}")
        print(f"[输出] 损伤帧: {frames_dir}")
        print(f"{'='*60}\n")
        
        return {
            'total_frames': frame_count,
            'damage_frames': damage_frame_count,
            'video_path': str(video_path),
            'frames_dir': str(frames_dir)
        }


def main():
    """主函数"""
    print("="*60)
    print("SVO分割模型测试工具")
    print("="*60)
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n[错误] 模型文件不存在: {MODEL_PATH}")
        print("请确认模型路径正确")
        return
    
    # 检查SVO文件
    if not os.path.exists(SVO_PATH):
        print(f"\n[错误] SVO文件不存在: {SVO_PATH}")
        print("请修改脚本中的 SVO_PATH 为正确的文件路径")
        print(f"\n可用的SVO文件:")
        svo_dir = os.path.dirname(SVO_PATH)
        if os.path.exists(svo_dir):
            for f in os.listdir(svo_dir):
                if f.endswith('.svo'):
                    print(f"  - {f}")
        return
    
    try:
        # 创建测试器
        tester = SVOSegmentationTester(MODEL_PATH, CONF_THRESHOLD)
        
        # 处理SVO
        results = tester.process_svo(SVO_PATH, OUTPUT_DIR)
        
        print("\n测试完成! 输出文件:")
        print(f"  视频: {results['video_path']}")
        print(f"  损伤帧: {results['frames_dir']}")
        
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
