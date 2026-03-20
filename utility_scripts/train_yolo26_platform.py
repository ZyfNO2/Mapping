# -*- coding: utf-8 -*-
"""
使用Ultralytics平台训练YOLO26 - 禁用离线模式尝试连接
"""

import os

# 设置API Key
os.environ['ULTRALYTICS_API_KEY'] = 'ul_06f17f6f1300f867af2edf68230f9fe258d21951'

# 尝试禁用代理（如果有的话）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from ultralytics import YOLO

def main():
    print("="*70)
    print("Training YOLO26 on Ultralytics Platform")
    print("="*70)
    
    try:
        # 加载YOLO26模型
        print("\nLoading YOLO26 model from platform...")
        model = YOLO('ul://ultralytics/yolo26/yolo26n-seg')
        
        # 训练
        print("\nStarting training with platform dataset...")
        model.train(
            data='ul://ad-astra/datasets/cracknet-images',
            epochs=100,
            batch=16,
            imgsz=640,
            project='ad-astra/heuristic-elephant-v3',
            name='train',
            verbose=True,
            workers=0
        )
        
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        print("\n建议：由于网络限制，请使用本地训练方案")
        print("运行: python train_local_crack.py")

if __name__ == '__main__':
    main()
