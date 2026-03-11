"""Check dataset labels"""
from pathlib import Path

# 检查标注文件
labels_dir = Path('trainData/crack_segmentation/labels/train')
label_files = list(labels_dir.glob('*.txt'))

print(f'找到 {len(label_files)} 个标注文件')
print('\n前5个标注文件内容:')
for i, label_file in enumerate(label_files[:5]):
    print(f'\n{label_file.name}:')
    with open(label_file, 'r') as f:
        content = f.read().strip()
        print(content if content else '(空文件)')

# 检查对应的图像
images_dir = Path('trainData/crack_segmentation/images/train')
print(f'\n\n图像数量: {len(list(images_dir.glob("*.jpg")))}')
print(f'标注数量: {len(label_files)}')

# 统计空文件
empty_count = 0
for label_file in label_files:
    with open(label_file, 'r') as f:
        if not f.read().strip():
            empty_count += 1

print(f'\n空标注文件: {empty_count}')
