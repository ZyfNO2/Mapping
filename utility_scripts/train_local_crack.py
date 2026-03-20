# -*- coding: utf-8 -*-
"""
使用本地裂缝数据集进行训练
"""

import os
from ultralytics import YOLO

def main():
    # 数据集配置路径
    data_yaml = r"G:\Zed\spatial mapping\crack_segmentation\dataset.yaml"

    print("="*70)
    print("Training with Local Crack Segmentation Dataset")
    print("="*70)
    print(f"Dataset config: {data_yaml}")

    # 加载预训练模型
    print("\nLoading YOLOv8n-seg model...")
    model = YOLO('yolov8n-seg.pt')

    # 训练
    print("\nStarting training...")
    model.train(
        data=data_yaml,
        epochs=100,
        batch=16,
        imgsz=640,
        project='crack_segmentation_training',
        name='exp',
        verbose=True,
        workers=0  # Windows下设置为0避免多进程问题
    )

    print("\nTraining completed!")
    print(f"Results saved to: runs/segment/crack_segmentation_training/exp")

if __name__ == '__main__':
    main()
