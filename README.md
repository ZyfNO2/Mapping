# ZED SDK - SVO Damage Detection V2

基于ZED SDK和YOLO分割模型的SVO文件损伤检测与点云标记系统。

## 功能特性

- **损伤检测**: 使用YOLO分割模型自动检测视频帧中的损伤区域
- **3D点云标记**: 将2D检测结果映射到3D点云，标记损伤位置
- **多输出格式**: 生成原始点云、标记点云、仅损伤区域点云等多种输出
- **点云后处理**: 密度过滤、统计滤波、法线估计、曲率过滤、聚类
- **损伤面积计算**: 基于Ball Pivoting算法计算损伤区域表面积

## 部署步骤

### 1. 克隆仓库
```bash
git clone <repository-url>
cd python
```

### 2. 创建Python环境
```bash
# 使用conda（推荐）
conda create -n zed python=3.10
conda activate zed

# 或使用venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

依赖包包括：
- `pyzed` - ZED SDK Python API
- `open3d` - 点云处理和可视化
- `ultralytics` - YOLO模型
- `opencv-python` - 图像处理
- `numpy`, `scipy`, `pandas` - 数值计算

### 4. 安装ZED SDK Python API
按照[官方指南](https://www.stereolabs.com/docs/app-development/python/install/)安装pyzed。

### 5. 准备YOLO模型
下载或训练YOLO分割模型，默认使用`seg.pt`：
```bash
# 将模型文件放在项目根目录
# 或使用自定义路径
```

## 使用方法

### 基础使用

处理SVO文件并进行损伤检测：
```bash
python svo_damage_detection_v2.py --svo "path\to\your.svo2"
```

### 高级选项

```bash
# 使用自定义模型
python svo_damage_detection_v2.py --svo "path\to\your.svo2" --model "path\to\best.pt"

# 调整置信度阈值（默认0.15）
python svo_damage_detection_v2.py --svo "path\to\your.svo2" --conf 0.2

# 保存检测图像
python svo_damage_detection_v2.py --svo "path\to\your.svo2" --save-images

# 可视化结果
python svo_damage_detection_v2.py --svo "path\to\your.svo2" --visualize
```

### 计算损伤面积

```bash
# 使用默认参数计算
python calculate_damage_area.py

# 指定damage_only.ply文件
python calculate_damage_area.py --ply "path\to\damage_only.ply"

# 调整重建参数
python calculate_damage_area.py --voxel-size 0.005 --radius 0.03

# 可视化重建表面
python calculate_damage_area.py --visualize
```

## 输出文件说明

运行`svo_damage_detection_v2.py`后，会在`data/damage_detection/<svo_name>/`目录下生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `{svo_name}_original.ply` | 原始重建点云（无标记，未处理） |
| `{svo_name}_original_processed.ply` | 原始点云（经过后处理） |
| `{svo_name}_marked.ply` | 带红色标记的完整点云（未处理） |
| `{svo_name}_marked_processed.ply` | 带标记点云（经过后处理） |
| `{svo_name}_damage_only.ply` | 仅包含损伤区域的点云（不处理） |
| `detection_images/` | 检测图像（如启用--save-images） |

## 技术架构

### 数据流流程

```
SVO文件输入
    ↓
ZED SDK读取帧
    ↓
YOLO分割检测 → 生成Mask
    ↓
提取3D点云（Camera坐标系）
    ↓
位姿变换 → World坐标系
    ↓
融合多帧损伤点
    ↓
提取完整点云（FusedPointCloud）
    ↓
标记损伤区域（红色）
    ↓
后处理（过滤、下采样、法线估计）
    ↓
多格式输出（PLY）
```

### 核心API

- `sl.Camera()` - ZED相机/SVO文件接口
- `sl.FusedPointCloud()` - 融合点云容器
- `zed.extract_whole_spatial_map()` - 提取完整空间映射
- `zed.retrieve_measure()` - 获取逐帧点云数据
- `YOLO()` - 损伤检测模型
- `o3d.geometry.PointCloud()` - Open3D点云处理

## 参数配置

### 检测参数

在`svo_damage_detection_v2.py`中可修改：

```python
CONF_THRESHOLD = 0.15          # 检测置信度阈值
MARK_COLOR = [1.0, 0.0, 0.0]   # 标记颜色（红色）
sample_step = 5                # 点云采样步长
threshold = 0.05               # 损伤点匹配阈值（米）
```

### 后处理参数

```python
# 密度过滤
density_radius = 0.06          # 搜索半径
min_density = 10               # 最小局部密度

# 统计滤波
nb_neighbors = 20              # 邻居点数
std_ratio = 2.0                # 标准差比率

# 体素下采样
voxel_size = 0.01              # 体素大小（米）

# 法线估计
radius = 0.1                   # 搜索半径
max_nn = 30                    # 最大邻居数
```

### 面积计算参数

```python
voxel_size = 0.01              # 下采样体素大小
ball_pivoting_search_radius = 0.05  # Ball Pivoting搜索半径
```

## 注意事项

1. **内存管理**: 大型SVO文件可能占用大量内存，建议设置`max_memory_usage`参数
2. **坐标系**: 输出点云使用世界坐标系（World Coordinate System）
3. **深度质量**: 使用`DEPTH_MODE.NEURAL`获得最佳深度估计质量
4. **GPU需求**: YOLO检测和深度估计需要NVIDIA GPU支持
5. **模型路径**: 确保YOLO模型路径正确，默认使用`seg.pt`

## 文件结构

```
python/
├── svo_damage_detection_v2.py    # 主程序：损伤检测与点云标记
├── calculate_damage_area.py      # 损伤面积计算
├── seg.pt                        # YOLO分割模型（需自行准备）
├── requirements.txt              # Python依赖
├── README.md                     # 本文件
└── data/
    └── damage_detection/         # 输出目录
        └── <svo_name>/
            ├── *_original.ply
            ├── *_marked.ply
            ├── *_damage_only.ply
            └── ...
```

## 许可证

本项目基于ZED SDK开发，遵循相应许可证协议。

---

*最后更新: 2026-03-29*
