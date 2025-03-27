# 隧道车辆多相机跟踪与重识别系统

基于AICITY2021_Track2_DMT项目，本系统实现了隧道内多摄像头视频中车辆的检测、跟踪、跨相机重识别，并将车辆位置映射到全局二维地图上。

## 系统功能

- 使用YOLOv8进行车辆检测
- 使用DeepSORT进行单相机内车辆跟踪
- 使用AICITY2021_Track2_DMT模型进行跨相机车辆重识别
- 将图像坐标映射到全局二维地图坐标
- 可视化跟踪结果和地图视图

## 系统架构

系统主要包含以下组件：

1. **车辆检测模块**：使用YOLOv8检测车辆
2. **车辆跟踪模块**：使用DeepSORT在单摄像头视图中跟踪车辆
3. **车辆重识别模块**：使用AICITY2021_Track2_DMT的ReID模型进行跨摄像头身份匹配
4. **坐标映射模块**：将图像坐标映射到全局二维地图坐标
5. **可视化模块**：显示跟踪结果和地图视图

## 安装说明

### 环境要求

- Python 3.7+
- CUDA 11.0+（如果使用GPU）
- 足够的磁盘空间（用于模型和数据）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/tunnel-vehicle-reid.git
   cd tunnel-vehicle-reid
   ```

2. 运行安装脚本：
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. 手动下载ReID模型权重：
   - 访问 [原项目模型权重](https://drive.google.com/drive/folders/1aCQmTbYQE-mq-07q86NIMLLZRc82mc5t?usp=sharing)
   - 下载 `resnext101_ibn_a_2.pth` 并放置到 `weights/reid/` 目录中

## 使用方法

1. 准备视频文件：
   - 将待处理的视频文件放入 `data/videos/` 目录
   - 视频文件应当按摄像头顺序命名，如 `camera1.mp4`, `camera2.mp4`, `camera3.mp4`

2. 配置摄像头参数：
   - 修改 `configs/camera_config.json` 文件，设置每个摄像头的内外参数
   - 配置应当包括摄像头位置、方向、内参矩阵、外参矩阵等

3. 运行系统：
   ```bash
   source venv/bin/activate
   python main.py --video_paths data/videos/camera1.mp4 data/videos/camera2.mp4 data/videos/camera3.mp4 --display --save_video
   ```

4. 命令行参数说明：
   - `--video_paths`: 输入视频路径列表（按摄像头顺序）
   - `--camera_config`: 摄像头配置文件路径（默认为 `configs/camera_config.json`）
   - `--reid_config`: ReID模型配置文件路径（默认为 `configs/reid_config.yaml`）
   - `--reid_weights`: ReID模型权重路径（默认为 `weights/reid/resnext101_ibn_a_2.pth`）
   - `--yolo_weights`: YOLO模型权重路径（默认为 `weights/yolo/yolov8m.pt`）
   - `--deepsort_weights`: DeepSORT模型权重路径
   - `--output_dir`: 输出目录（默认为 `output`）
   - `--device`: 运行设备（`cuda` 或 `cpu`，默认为 `cuda`）
   - `--display`: 是否显示结果（如不指定则不显示）
   - `--save_video`: 是否保存结果视频（如不指定则不保存）

## 输出结果

运行系统后，将会产生以下输出：

1. 实时显示窗口（如使用 `--display` 选项）：
   - 左侧为摄像头视图，每个车辆带有ID标签
   - 右侧为二维地图视图，显示车辆在地图上的位置和轨迹

2. 输出视频（如使用 `--save_video` 选项）：
   - 每个摄像头的单独输出视频：`output/camera_1_output.mp4` 等
   - 所有视图的合并视频：`output/combined_output.mp4`

## 系统自定义

### 自定义检测模型

如果需要使用不同的YOLOv8模型，可以下载其他模型权重并通过命令行参数指定：

```bash
python main.py --video_paths ... --yolo_weights weights/yolo/yolov8l.pt
```

### 自定义ReID模型

如需使用原项目中的其他ReID模型，需要：

1. 修改 `configs/reid_config.yaml` 中的模型设置
2. 下载对应的模型权重并放置到 `weights/reid/` 目录中
3. 通过命令行参数指定配置和权重：

```bash
python main.py --video_paths ... --reid_config configs/my_reid_config.yaml --reid_weights weights/reid/my_model.pth
```

### 自定义摄像头配置

要调整摄像头的位置和参数，编辑 `configs/camera_config.json` 文件：

1. 每个摄像头的内参矩阵（`intrinsic_matrix`）
2. 旋转矩阵（`rotation_matrix`）和平移向量（`translation_vector`）
3. 在地图上的位置（`position`）和朝向（`direction`）

## 项目来源

本项目基于以下开源项目：

- [AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)：车辆重识别模型
- [YOLOv8](https://github.com/ultralytics/ultralytics)：车辆检测模型
- [Deep SORT](https://github.com/ZQPei/deep_sort_pytorch)：车辆跟踪算法

## 注意事项

- 系统性能依赖于硬件配置，特别是GPU性能
- 坐标映射精度依赖于摄像头标定的准确性
- 在使用自己的视频时，需要调整摄像头参数以获得准确的定位结果
