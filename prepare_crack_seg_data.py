"""
Crack Segmentation Dataset Preparation
Downloads images from cracknet-images.ndjson and converts to YOLO segmentation format
"""

import json
import os
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np


def download_image(url, save_path):
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def convert_segments_to_yolo(segments, width, height):
    """
    Convert segments format to YOLO format
    segments: [[class_id, x1, y1, x2, y2, ...], ...]
    YOLO format: class_id x1 y1 x2 y2 ... (normalized 0-1)
    """
    yolo_annotations = []
    
    for seg in segments:
        if len(seg) < 3:
            continue
            
        class_id = int(seg[0])
        points = seg[1:]
        
        # Normalize points
        normalized_points = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                x = points[i] / width
                y = points[i + 1] / height
                normalized_points.extend([x, y])
        
        if normalized_points:
            yolo_annotations.append([class_id] + normalized_points)
    
    return yolo_annotations


def prepare_dataset(ndjson_path, output_dir):
    """
    Prepare dataset from ndjson file
    """
    output_dir = Path(output_dir)
    
    # Create directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    (images_dir / 'train').mkdir(parents=True, exist_ok=True)
    (images_dir / 'val').mkdir(parents=True, exist_ok=True)
    (labels_dir / 'train').mkdir(parents=True, exist_ok=True)
    (labels_dir / 'val').mkdir(parents=True, exist_ok=True)
    
    # Read ndjson file
    with open(ndjson_path, 'r') as f:
        lines = f.readlines()
    
    # Parse dataset info
    dataset_info = json.loads(lines[0])
    print(f"Dataset: {dataset_info.get('name', 'Unknown')}")
    print(f"Task: {dataset_info.get('task', 'Unknown')}")
    print(f"Total entries: {len(lines) - 1}")
    
    # Process images
    train_count = 0
    val_count = 0
    failed_downloads = []
    
    total = len(lines) - 1
    for i, line in enumerate(lines[1:]):
        if i % 10 == 0:
            print(f"Processing {i}/{total} images...")
        try:
            data = json.loads(line)
            
            if data.get('type') != 'image':
                continue
            
            # Get image info
            filename = data['file']
            url = data['url']
            width = data['width']
            height = data['height']
            split = data.get('split', 'train')
            annotations = data.get('annotations', {})
            
            # Determine output directories
            if split == 'val':
                img_dir = images_dir / 'val'
                lbl_dir = labels_dir / 'val'
                val_count += 1
            else:
                img_dir = images_dir / 'train'
                lbl_dir = labels_dir / 'train'
                train_count += 1
            
            # Download image
            img_path = img_dir / filename
            if not img_path.exists():
                success = download_image(url, img_path)
                if not success:
                    failed_downloads.append(filename)
                    continue
            
            # Convert and save annotations
            segments = annotations.get('segments', [])
            if segments:
                yolo_annotations = convert_segments_to_yolo(segments, width, height)
                
                # Save YOLO format annotation
                label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = lbl_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in yolo_annotations:
                        line = ' '.join([str(x) for x in ann])
                        f.write(line + '\n')
                        
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    
    print(f"\nDataset preparation complete!")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"Failed files: {failed_downloads[:5]}...")  # Show first 5
    
    return train_count, val_count


def create_dataset_yaml(output_dir):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# Crack Segmentation Dataset
path: {output_dir}  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
names:
  0: crack

# Number of classes
nc: 1
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset YAML created: {yaml_path}")


if __name__ == "__main__":
    # Configuration
    NDJSON_PATH = "trainData/cracknet-images.ndjson"
    OUTPUT_DIR = "trainData/crack_segmentation"
    
    # Prepare dataset
    train_count, val_count = prepare_dataset(NDJSON_PATH, OUTPUT_DIR)
    
    # Create YAML file
    create_dataset_yaml(OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Verify downloaded images in trainData/crack_segmentation/images/")
    print("2. Verify labels in trainData/crack_segmentation/labels/")
    print("3. Train YOLO model with:")
    print("   python train_crack_seg.py")
