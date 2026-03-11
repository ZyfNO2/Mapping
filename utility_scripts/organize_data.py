"""
整理data文件夹结构
将不同视频的结果放在不同的子文件夹中
"""
import shutil
from pathlib import Path
import os

def organize_data_folder():
    """整理data文件夹"""
    
    data_path = Path("data")
    
    print("="*70)
    print("Organizing Data Folder")
    print("="*70)
    
    # 检查当前data文件夹内容
    if not data_path.exists():
        print("Creating data folder...")
        data_path.mkdir(parents=True, exist_ok=True)
        return
    
    # 列出当前内容
    print("\nCurrent data folder contents:")
    items = list(data_path.iterdir())
    for item in items:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name:<50} {size_mb:>8.2f} MB")
        else:
            print(f"  📁 {item.name}/")
    
    # 检查是否已经有旧视频的数据
    old_svo_name = "HD2K_SN36245620_15-16-49"
    new_svo_name = "HD2K_SN36245620_15-17-12"
    
    # 创建旧视频的文件夹
    old_folder = data_path / old_svo_name
    new_folder = data_path / new_svo_name
    
    print(f"\nCreating folders:")
    print(f"  📁 {old_folder}/")
    print(f"  📁 {new_folder}/")
    
    old_folder.mkdir(exist_ok=True)
    new_folder.mkdir(exist_ok=True)
    
    # 移动旧文件到旧视频文件夹
    files_to_move = [
        "point_cloud_gen_high_quality.ply",
        "point_cloud_processed.ply",
        "point_cloud_downsampled.ply",
        "point_cloud_with_normals.ply",
        "point_cloud_final.ply",
        "point_cloud.csv",
        "point_cloud.obj"
    ]
    
    moved_count = 0
    for filename in files_to_move:
        src = data_path / filename
        if src.exists():
            dst = old_folder / filename
            print(f"\n  Moving: {filename}")
            print(f"    From: {src}")
            print(f"    To:   {dst}")
            shutil.move(str(src), str(dst))
            moved_count += 1
    
    print(f"\n" + "="*70)
    print(f"Organization complete!")
    print(f"  Moved {moved_count} files to {old_folder}/")
    print(f"  New video data will be saved to {new_folder}/")
    print("="*70)
    
    # 显示新的文件夹结构
    print("\nNew data folder structure:")
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # 列出子文件夹内容
            sub_items = list(item.iterdir())
            for sub_item in sorted(sub_items)[:5]:  # 最多显示5个
                if sub_item.is_file():
                    size_mb = sub_item.stat().st_size / (1024 * 1024)
                    print(f"      📄 {sub_item.name:<40} {size_mb:>8.2f} MB")
            if len(sub_items) > 5:
                print(f"      ... and {len(sub_items) - 5} more files")

if __name__ == "__main__":
    organize_data_folder()
