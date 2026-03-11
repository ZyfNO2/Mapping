"""Find all model files (.pt) in the project"""
from pathlib import Path
import os

def find_model_files():
    base_path = Path("G:/Zed/spatial mapping/python")
    
    print("="*70)
    print("Searching for model files (.pt)")
    print("="*70)
    
    # Find all .pt files
    pt_files = list(base_path.rglob("*.pt"))
    
    if not pt_files:
        print("No .pt files found!")
        return
    
    print(f"\nFound {len(pt_files)} model files:\n")
    
    # Group by directory
    files_by_dir = {}
    for pt_file in pt_files:
        dir_path = pt_file.parent
        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        files_by_dir[dir_path].append(pt_file)
    
    # Print organized results
    for dir_path, files in sorted(files_by_dir.items()):
        print(f"\n📁 {dir_path}")
        print("-" * 70)
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  📄 {f.name:<50} {size_mb:>8.2f} MB")
    
    print("\n" + "="*70)
    print("Summary:")
    print(f"Total model files: {len(pt_files)}")
    print("="*70)

if __name__ == "__main__":
    find_model_files()
