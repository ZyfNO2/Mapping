# -*- coding: utf-8 -*-
"""
本地训练脚本 - 禁用Ultralytics平台连接
"""

import os
os.environ['ULTRALYTICS_API_KEY'] = 'ul_06f17f6f1300f867af2edf68230f9fe258d21951'
os.environ['ULTRALYTICS_OFFLINE'] = '1'  # 禁用平台连接

from ultralytics import YOLO

# 加载模型
print("Loading YOLOv8n-seg model...")
model = YOLO('yolov8n-seg.pt')

# 训练
print("Starting training...")
model.train(
    data='ul://ad-astra/datasets/cracknet-images',
    epochs=100,
    batch=16,
    imgsz=640,
    project='ad-astra/heuristic-elephant-v3',
    verbose=True
)

print("Training completed!")
