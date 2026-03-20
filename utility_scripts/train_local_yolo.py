# -*- coding: utf-8 -*-
"""
本地训练YOLOv8分割模型 - 使用已下载的cracknet数据集
"""

import os
from ultralytics import YOLO

def main():
    print("="*70)
    print("本地训练YOLOv8分割模型")
    print("="*70)
    
    # 数据集配置路径
    data_yaml = r'G:\Zed\spatial mapping\python\datasets\cracknet-images\data.yaml'
    
    print(f"\n数据集配置: {data_yaml}")
    
    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        print(f"[错误] 找不到数据集配置文件: {data_yaml}")
        return
    
    # 加载预训练的YOLOv8n-seg模型
    print("\n加载YOLOv8n-seg模型...")
    model = YOLO('yolov8n-seg.pt')
    
    # 训练配置
    print("\n开始训练...")
    print("配置:")
    print(f"  - Epochs: 100")
    print(f"  - Batch size: 16")
    print(f"  - Image size: 640")
    print(f"  - 项目: cracknet_training")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        batch=16,
        imgsz=640,
        project='cracknet_training',
        name='yolov8n_seg',
        verbose=True,
        workers=0,  # Windows下避免多进程问题
        patience=20,  # 早停耐心值
        save=True,  # 保存最佳模型
        device=0  # 使用GPU，如果没有则自动使用CPU
    )
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"\n最佳模型保存在: {results.best}")
    
    # 验证模型
    print("\n开始验证...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
