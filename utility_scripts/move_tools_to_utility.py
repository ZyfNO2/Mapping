# -*- coding: utf-8 -*-
"""
将工具脚本移动到utility_scripts目录
"""

import os
import shutil
from pathlib import Path


def move_tool_scripts():
    """移动工具脚本到utility_scripts目录"""
    
    # 当前目录
    current_dir = Path(__file__).parent
    utility_dir = current_dir / "utility_scripts"
    
    # 确保utility_scripts目录存在
    utility_dir.mkdir(exist_ok=True)
    
    # 要移动的工具脚本列表
    tool_scripts = [
        "check_pointcloud.py",
        "debug_matching.py", 
        "generate_comparison_image.py",
        "test_mask_extraction.py",
        "visualize_mask_pointcloud.py",
        "mask_point_cloud_marker.py",  # 旧版本
    ]
    
    print("="*70)
    print("Moving Tool Scripts to utility_scripts/")
    print("="*70)
    
    moved_count = 0
    for script_name in tool_scripts:
        source = current_dir / script_name
        destination = utility_dir / script_name
        
        if source.exists():
            if destination.exists():
                print(f"[Warning] {script_name} already exists in utility_scripts/, skipping")
            else:
                shutil.move(str(source), str(destination))
                print(f"[Moved] {script_name}")
                moved_count += 1
        else:
            print(f"[Not Found] {script_name}")
    
    print("\n" + "="*70)
    print(f"Moved {moved_count} scripts to utility_scripts/")
    print("="*70)
    
    # 列出utility_scripts目录内容
    print("\nCurrent utility_scripts/ contents:")
    for item in sorted(utility_dir.glob("*.py")):
        print(f"  - {item.name}")


if __name__ == "__main__":
    move_tool_scripts()
