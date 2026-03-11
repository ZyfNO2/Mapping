"""
SVO文件裂缝分割处理 - 过滤版本
只保存检测到裂缝的帧，输出原图、分割图和mask JSON

处理流程：
1. 读取SVO文件 (HD2K_SN36245620_15-17-12)
2. 加载YOLO分割模型 (best.pt)，低置信度阈值
3. 对每一帧进行裂缝分割
4. 只保存检测到裂缝的帧：
   - 原图 (original/)
   - 带mask的图 (segmented/)
   - mask JSON文件 (masks/)
"""
import sys
import time
import cv2
import json
import numpy as np
import pyzed.sl as sl
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def mask_to_json(mask, confidence, box):
    """
    将mask转换为JSON格式
    
    Args:
        mask: 二值mask (H, W)
        confidence: 置信度
        box: 边界框 [x1, y1, x2, y2]
    
    Returns:
        dict: mask信息
    """
    # 获取mask的轮廓点（简化表示）
    mask_bool = mask > 0.5
    
    # 计算mask面积
    area = int(np.sum(mask_bool))
    
    # 获取mask的边界框
    y_indices, x_indices = np.where(mask_bool)
    if len(y_indices) > 0:
        bbox = {
            "x1": int(np.min(x_indices)),
            "y1": int(np.min(y_indices)),
            "x2": int(np.max(x_indices)),
            "y2": int(np.max(y_indices))
        }
    else:
        bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
    
    # 将mask转换为RLE格式（简化）
    # 这里我们保存mask的稀疏表示
    mask_data = {
        "shape": list(mask.shape),
        "area": area,
        "bbox": bbox,
        "confidence": float(confidence),
        "detection_box": [float(x) for x in box]
    }
    
    return mask_data


def apply_mask_overlay(image, mask, alpha=0.5, color=(0, 0, 255)):
    """
    将分割mask叠加到图像上
    """
    overlay = image.copy()
    
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    mask_bool = mask > 0.5
    colored_mask = np.zeros_like(image)
    colored_mask[mask_bool] = color
    
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha,
        colored_mask[mask_bool], alpha, 0
    )
    
    return overlay


def process_svo_segmentation_filtered(
    svo_path,
    model_path,
    output_dir="output_segmentation_filtered",
    conf_threshold=0.15,  # 降低置信度阈值
    save_original=True,
    save_segmented=True,
    save_json=True
):
    """
    处理SVO文件进行裂缝分割（过滤版本）
    """
    print("="*70)
    print("SVO Crack Segmentation - Filtered Version")
    print("="*70)
    print(f"Confidence threshold: {conf_threshold}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    original_dir = output_path / "original"
    segmented_dir = output_path / "segmented"
    masks_dir = output_path / "masks"
    
    original_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 获取总帧数
    print("Counting frames...")
    total_frames = 0
    zed.set_svo_position(0)
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        total_frames += 1
    zed.set_svo_position(0)
    print(f"Total frames: {total_frames}")
    
    # 处理每一帧
    print("\nProcessing frames...")
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            grab_result = zed.grab()
            if grab_result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            
            if grab_result != sl.ERROR_CODE.SUCCESS:
                continue
            
            # 获取图像
            image_zed = sl.Mat()
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            
            # 转换为RGB和BGR
            if frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = frame.copy()
            
            # YOLO分割（使用低置信度阈值）
            results = model(frame_rgb, verbose=False, conf=conf_threshold)
            
            # 检查是否检测到裂缝
            has_crack = False
            masks_data = []
            annotated_frame = frame_bgr.copy()
            
            if results[0].masks is not None and len(results[0].masks) > 0:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                # 过滤低置信度的检测结果
                valid_indices = confs >= conf_threshold
                
                if np.any(valid_indices):
                    has_crack = True
                    
                    for i, (mask, box, conf) in enumerate(zip(masks, boxes, confs)):
                        if conf < conf_threshold:
                            continue
                        
                        # 调整mask尺寸
                        mask_resized = cv2.resize(
                            mask, 
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        # 叠加mask
                        annotated_frame = apply_mask_overlay(
                            annotated_frame, 
                            mask_resized, 
                            alpha=0.4, 
                            color=(0, 0, 255)
                        )
                        
                        # 画框和标签
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"crack: {conf:.2f}"
                        cv2.putText(
                            annotated_frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2
                        )
                        
                        # 保存mask数据
                        mask_json = mask_to_json(mask_resized, conf, box)
                        masks_data.append(mask_json)
            
            # 只保存检测到裂缝的帧
            if has_crack:
                # 保存原图
                if save_original:
                    original_path = original_dir / f"frame_{saved_count:06d}_original.jpg"
                    cv2.imwrite(str(original_path), frame_bgr)
                
                # 保存分割图
                if save_segmented:
                    # 添加帧号和裂缝数量
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_count} | Cracks: {len(masks_data)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2
                    )
                    segmented_path = segmented_dir / f"frame_{saved_count:06d}_segmented.jpg"
                    cv2.imwrite(str(segmented_path), annotated_frame)
                
                # 保存mask JSON
                if save_json:
                    json_data = {
                        "frame_id": frame_count,
                        "saved_id": saved_count,
                        "num_cracks": len(masks_data),
                        "masks": masks_data
                    }
                    json_path = masks_dir / f"frame_{saved_count:06d}_masks.json"
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({"Saved": saved_count})
    
    # 释放资源
    zed.close()
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with cracks saved: {saved_count}")
    print(f"Save rate: {saved_count/frame_count*100:.1f}%")
    if save_original:
        print(f"Original images: {original_dir}")
    if save_segmented:
        print(f"Segmented images: {segmented_dir}")
    if save_json:
        print(f"Mask JSON files: {masks_dir}")
    print("="*70)


def main():
    """主函数"""
    # 配置参数 - 使用新的SVO文件
    SVO_PATH = "C:\\Users\\ZYF\\Documents\\ZED\\HD2K_SN36245620_15-17-12.svo2"
    MODEL_PATH = "runs/segment/ad-astra/heuristic-elephant/train/weights/best.pt"
    OUTPUT_DIR = "output_segmentation_filtered"
    CONF_THRESHOLD = 0.15  # 降低置信度阈值，检测更多潜在裂缝
    
    # 检查文件是否存在
    if not Path(SVO_PATH).exists():
        print(f"Error: SVO file not found: {SVO_PATH}")
        print("Searching for available SVO files...")
        
        # 搜索可用的SVO文件
        svo_files = list(Path("C:\\Users\\ZYF\\Documents\\ZED").glob("*.svo2"))
        if svo_files:
            print("\nAvailable SVO files:")
            for i, svo in enumerate(svo_files, 1):
                print(f"{i}. {svo.name}")
            SVO_PATH = str(svo_files[0])
            print(f"\nUsing: {SVO_PATH}")
        else:
            print("No SVO files found!")
            return
    
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Searching for available models...")
        
        best_models = list(Path(".").rglob("best.pt"))
        if best_models:
            print("\nAvailable best.pt models:")
            for i, model in enumerate(best_models, 1):
                print(f"{i}. {model}")
            MODEL_PATH = str(best_models[0])
            print(f"\nUsing: {MODEL_PATH}")
        else:
            print("No best.pt found! Please train a model first.")
            return
    
    # 运行处理
    process_svo_segmentation_filtered(
        svo_path=SVO_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        save_original=True,
        save_segmented=True,
        save_json=True
    )


if __name__ == "__main__":
    main()
