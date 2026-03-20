# -*- coding: utf-8 -*-
"""
测试 mask 提取功能（无需 ZED SDK）

这个脚本用于验证 mask 提取逻辑是否正确，不需要 ZED SDK
"""

import cv2
import numpy as np
import json
import os


def extract_mask_from_segmented_image(segmented_image_path: str, red_threshold: int = 100):
    """
    从分割后的图像中提取mask（基于红色标记区域）
    """
    try:
        # 读取分割图像
        segmented = cv2.imread(segmented_image_path)
        if segmented is None:
            print(f"[Error] Failed to load segmented image: {segmented_image_path}")
            return None
        
        print(f"[Info] Loaded segmented image with shape: {segmented.shape}")
        
        # 转换到HSV色彩空间，更容易提取红色
        hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        
        # 红色的HSV范围（有两个范围，因为红色在HSV中跨越0度）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建红色mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        mask_area = np.sum(red_mask > 0)
        print(f"[Info] Extracted mask area: {mask_area} pixels")
        
        return red_mask
        
    except Exception as e:
        print(f"[Error] Failed to extract mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mask_extraction():
    """测试 mask 提取功能"""
    print("="*70)
    print("Testing Mask Extraction")
    print("="*70)
    
    # 测试文件路径
    frame_id = 33
    segmentation_dir = r"G:\Zed\spatial mapping\output_segmentation_filtered"
    
    mask_json_path = os.path.join(segmentation_dir, 'masks', f'frame_{frame_id:06d}_masks.json')
    segmented_image_path = os.path.join(segmentation_dir, 'segmented', f'frame_{frame_id:06d}_segmented.jpg')
    
    print(f"\n[Info] Testing frame {frame_id}")
    print(f"[Info] Mask JSON: {mask_json_path}")
    print(f"[Info] Segmented image: {segmented_image_path}")
    
    # 检查文件是否存在
    if not os.path.exists(mask_json_path):
        print(f"[Error] Mask JSON not found!")
        return False
    
    if not os.path.exists(segmented_image_path):
        print(f"[Error] Segmented image not found!")
        return False
    
    print("[Info] All files found!")
    
    # 加载 mask JSON
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    
    print(f"\n[Info] Mask JSON content:")
    print(f"  - frame_id: {mask_data.get('frame_id')}")
    print(f"  - saved_id: {mask_data.get('saved_id')}")
    print(f"  - num_cracks: {mask_data.get('num_cracks')}")
    
    if mask_data.get('masks'):
        for i, mask_info in enumerate(mask_data['masks']):
            print(f"\n  Mask {i+1}:")
            print(f"    - shape: {mask_info.get('shape')}")
            print(f"    - area: {mask_info.get('area')}")
            print(f"    - bbox: {mask_info.get('bbox')}")
            print(f"    - confidence: {mask_info.get('confidence'):.4f}")
    
    # 提取 mask
    print("\n[Info] Extracting mask from segmented image...")
    mask = extract_mask_from_segmented_image(segmented_image_path)
    
    if mask is not None:
        print("[Success] Mask extraction works!")
        
        # 保存 mask 图像用于验证
        output_dir = "data/mask_test"
        os.makedirs(output_dir, exist_ok=True)
        
        mask_output_path = os.path.join(output_dir, f"frame_{frame_id:06d}_extracted_mask.png")
        cv2.imwrite(mask_output_path, mask)
        print(f"[Info] Saved extracted mask to: {mask_output_path}")
        
        # 创建可视化图像
        segmented = cv2.imread(segmented_image_path)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[mask > 0] = [0, 255, 0]  # 绿色标记 mask 区域
        
        # 叠加显示
        overlay = cv2.addWeighted(segmented, 0.7, mask_colored, 0.3, 0)
        
        overlay_path = os.path.join(output_dir, f"frame_{frame_id:06d}_mask_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"[Info] Saved overlay image to: {overlay_path}")
        
        return True
    else:
        print("[Error] Mask extraction failed!")
        return False


if __name__ == "__main__":
    success = test_mask_extraction()
    
    if success:
        print("\n" + "="*70)
        print("All tests passed! ✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Tests failed! ✗")
        print("="*70)
