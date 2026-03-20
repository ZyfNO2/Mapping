# -*- coding: utf-8 -*-
"""
批量处理SVO文件脚本
使用svo_damage_detection_v2.py处理指定目录下的所有SVO文件

使用方法：
    python batch_process_svo.py
    python batch_process_svo.py --svo-dir "C:/Users/ZYF/Documents/ZED"
    python batch_process_svo.py --model "path/to/model.pt"
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime


# 默认配置
DEFAULT_SVO_DIR = r"C:\Users\ZYF\Documents\ZED"
DEFAULT_MODEL_PATH = r"G:\Zed\spatial mapping\python\seg.pt"
DEFAULT_OUTPUT_DIR = "data/damage_detection"
PYTHON_EXE = r"G:\Anaconda\envs\zed\python.exe"


def get_svo_files(svo_dir: str) -> list:
    """获取目录下所有SVO文件"""
    svo_files = []
    if os.path.exists(svo_dir):
        for file in os.listdir(svo_dir):
            if file.endswith('.svo') or file.endswith('.svo2'):
                svo_files.append(os.path.join(svo_dir, file))
    return sorted(svo_files)


def process_svo_file(svo_path: str, model_path: str, output_dir: str) -> bool:
    """
    处理单个SVO文件
    
    Returns:
        bool: 处理是否成功
    """
    svo_name = Path(svo_path).stem
    print("\n" + "="*70)
    print(f"Processing: {svo_name}")
    print("="*70)
    
    # 构建命令
    cmd = [
        PYTHON_EXE,
        "svo_damage_detection_v2.py",
        "--svo", svo_path,
        "--model", model_path,
        "--output-dir", output_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行处理脚本
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print(f"\n✅ {svo_name} processed successfully!")
            return True
        else:
            print(f"\n❌ {svo_name} failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error processing {svo_name}: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Batch process SVO files with damage detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_svo.py
  python batch_process_svo.py --svo-dir "C:\\Users\\ZYF\\Documents\\ZED"
  python batch_process_svo.py --model "path\\to\\model.pt"
  python batch_process_svo.py --skip-existing  # 跳过已处理的文件
        """
    )
    
    parser.add_argument(
        '--svo-dir',
        type=str,
        default=DEFAULT_SVO_DIR,
        help=f'SVO files directory (default: {DEFAULT_SVO_DIR})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Model path (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip SVO files that have already been processed'
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.svo_dir):
        print(f"[Error] SVO directory not found: {args.svo_dir}")
        return
    
    # 检查模型是否存在
    if not os.path.exists(args.model):
        print(f"[Error] Model file not found: {args.model}")
        return
    
    # 获取所有SVO文件
    svo_files = get_svo_files(args.svo_dir)
    
    if not svo_files:
        print(f"[Error] No SVO files found in: {args.svo_dir}")
        return
    
    print("="*70)
    print("Batch SVO Processing")
    print("="*70)
    print(f"SVO Directory: {args.svo_dir}")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total SVO files found: {len(svo_files)}")
    print("="*70)
    
    # 打印文件列表
    print("\nSVO files to process:")
    for i, svo_file in enumerate(svo_files, 1):
        file_size = os.path.getsize(svo_file) / (1024*1024)  # MB
        print(f"  {i}. {Path(svo_file).name} ({file_size:.2f} MB)")
    
    # 如果需要跳过已处理的文件
    if args.skip_existing:
        print("\n[Info] Checking for existing outputs...")
        svo_files_to_process = []
        for svo_file in svo_files:
            svo_name = Path(svo_file).stem
            output_path = os.path.join(args.output_dir, svo_name)
            if os.path.exists(output_path):
                print(f"  - Skipping {svo_name} (already processed)")
            else:
                svo_files_to_process.append(svo_file)
        svo_files = svo_files_to_process
        print(f"[Info] {len(svo_files)} files to process")
    
    if not svo_files:
        print("\n[Info] All files have been processed. Nothing to do.")
        return
    
    # 批量处理
    print("\n" + "="*70)
    print("Starting batch processing...")
    print("="*70)
    
    start_time = time.time()
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for i, svo_file in enumerate(svo_files, 1):
        print(f"\n\n[{i}/{len(svo_files)}] Processing file...")
        
        file_start = time.time()
        success = process_svo_file(svo_file, args.model, args.output_dir)
        file_duration = time.time() - file_start
        
        if success:
            success_count += 1
            print(f"⏱️  Duration: {file_duration/60:.1f} minutes")
        else:
            failed_count += 1
            failed_files.append(Path(svo_file).name)
        
        # 显示进度
        progress = (i / len(svo_files)) * 100
        print(f"\n📊 Overall Progress: {i}/{len(svo_files)} ({progress:.1f}%)")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {failed_count}")
    
    # 总结
    total_duration = time.time() - start_time
    
    print("\n" + "="*70)
    print("Batch Processing Complete!")
    print("="*70)
    print(f"Total files: {len(svo_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
