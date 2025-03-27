import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
from threading import Thread
from queue import Queue

# 导入项目组件
from models.detection.yolo_detector import VehicleDetector
from models.tracking.deep_sort_tracker import VehicleTracker
from models.reid.reid_model import VehicleReID
from tracking.reid_matcher import ReIDMatcher
from utils.mapping_utils import CoordinateMapper
from utils.visualization import Visualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='隧道车辆多相机跟踪与重识别系统')
    
    parser.add_argument('--video_paths', nargs='+', required=True,
                      help='输入视频路径，按摄像头顺序排列')
    parser.add_argument('--camera_config', type=str, default='configs/camera_config.json',
                      help='摄像头配置文件路径')
    parser.add_argument('--reid_config', type=str, default='configs/reid_config.yaml',
                      help='ReID模型配置文件路径')
    parser.add_argument('--reid_weights', type=str, default='weights/reid/resnext101_ibn_a_2.pth',
                      help='ReID模型权重路径')
    parser.add_argument('--yolo_weights', type=str, default='weights/yolo/yolov8s.pt',
                      help='YOLO模型权重路径')
    parser.add_argument('--deepsort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                      help='DeepSORT模型权重路径')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备 (cuda 或 cpu)')
    parser.add_argument('--display', action='store_true',
                      help='是否显示结果')
    parser.add_argument('--save_video', action='store_true',
                      help='是否保存结果视频')
    
    return parser.parse_args()

def process_camera_feed(video_path, camera_id, detector, tracker, reid_model, coordinator, args,
                       frames_queue, results_queue):
    """
    处理单个摄像头的视频流
    
    参数:
        video_path: 视频路径
        camera_id: 摄像头ID
        detector: 车辆检测器
        tracker: 车辆跟踪器
        reid_model: ReID模型
        coordinator: 坐标映射器
        args: 命令行参数
        frames_queue: 帧队列
        results_queue: 结果队列
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果需要保存视频
    output_video = None
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"camera_{camera_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 检测车辆
        detections = detector.detect(frame)
        
        # 跟踪车辆
        tracks = tracker.update(frame, detections)
        
        # 将底部中心点映射到地图坐标
        vehicle_positions = []
        vehicle_patches = []
        vehicle_track_ids = []
        
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            # 计算底部中心点
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            
            # 计算在地图上的位置
            map_pos = coordinator.image_to_ground(camera_id, (bottom_center_x, bottom_center_y))  # 不再需要高度参数
            
            # 记录位置信息
            vehicle_positions.append((track_id, map_pos))
            
            # 提取车辆图像块
            vehicle_patch = frame[int(y1):int(y2), int(x1):int(x2)]
            vehicle_patches.append(vehicle_patch)
            vehicle_track_ids.append(track_id)
        
        # 将结果放入队列
        results_queue.put({
            'camera_id': camera_id,
            'frame': frame,
            'tracks': tracks,
            'positions': vehicle_positions,
            'patches': vehicle_patches,
            'track_ids': vehicle_track_ids,
            'frame_count': frame_count
        })
        
        # 将帧放入队列
        frames_queue.put({
            'camera_id': camera_id,
            'frame': frame.copy(),
            'frame_count': frame_count
        })
        
        # 控制处理速度
        time.sleep(1.0 / fps)
    
    # 释放资源
    cap.release()
    if output_video is not None:
        output_video.release()
        
    # 通知主线程该摄像头处理完毕
    results_queue.put({
        'camera_id': camera_id,
        'finished': True
    })

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输入视频数量正确
    if len(args.video_paths) < 2:
        print("错误：至少需要两个摄像头视频")
        return
        
    # 加载摄像头配置
    with open(args.camera_config, 'r') as f:
        camera_config = json.load(f)
        
    # 创建组件
    detector = VehicleDetector(model_path=args.yolo_weights, device=args.device)
    trackers = {}
    for i in range(len(args.video_paths)):
        camera_id = str(i + 1)
        trackers[camera_id] = VehicleTracker(model_weights=args.deepsort_weights)
        
    reid_model = VehicleReID(args.reid_config, args.reid_weights, device=args.device)
    coordinator = CoordinateMapper(args.camera_config)
    
    # 提取摄像头位置用于可视化
    camera_positions = {}
    for camera_id, config in camera_config.items():
        if camera_id in ['ground_normal', 'ground_point']:
            continue
        camera_positions[camera_id] = config.get('position', (0, 0))
        
    visualizer = Visualizer(map_size=(800, 100), camera_positions=camera_positions)
    
    # 创建ReID匹配器
    reid_matcher = ReIDMatcher(reid_model, distance_threshold=0.7, match_interval=5)
    
    # 创建队列
    frames_queue = Queue()
    results_queue = Queue()
    
    # 启动摄像头处理线程
    camera_threads = []
    for i, video_path in enumerate(args.video_paths):
        camera_id = str(i + 1)
        thread = Thread(target=process_camera_feed,
                        args=(video_path, camera_id, detector, trackers[camera_id], 
                              reid_model, coordinator, args, frames_queue, results_queue))
        thread.daemon = True
        thread.start()
        camera_threads.append(thread)
    
    # 主循环
    active_cameras = len(args.video_paths)
    frame_buffer = {}
    
    # 如果需要保存合并视频
    output_video = None
    
    try:
        while active_cameras > 0:
            # 处理结果队列
            if not results_queue.empty():
                result = results_queue.get()
                
                # 检查是否有摄像头处理完毕
                if result.get('finished', False):
                    active_cameras -= 1
                    continue
                
                camera_id = result['camera_id']
                vehicle_patches = result['patches']
                vehicle_track_ids = result['track_ids']
                
                # 更新ReID特征
                if vehicle_patches:
                    reid_matcher.update_features(camera_id, vehicle_track_ids, vehicle_patches)
                
                # 每5帧执行一次跨摄像头匹配
                if result['frame_count'] % 5 == 0:
                    reid_matcher.match_across_cameras()
            
            # 处理帧队列
            if not frames_queue.empty():
                frame_data = frames_queue.get()
                camera_id = frame_data['camera_id']
                frame = frame_data['frame']
                frame_count = frame_data['frame_count']
                
                # 存储帧以同步显示
                frame_buffer[camera_id] = frame
                
                # 如果所有摄像头的帧都准备好了
                if len(frame_buffer) == len(args.video_paths):
                    # 收集所有车辆的地图位置
                    map_positions = []
                    for cam_id, frame in frame_buffer.items():
                        # 获取该摄像头的跟踪结果
                        if not results_queue.empty():
                            result = results_queue.get()
                            while result.get('camera_id') != cam_id and not result.get('finished', False):
                                results_queue.put(result)  # 放回队列
                                result = results_queue.get()
                            
                            if not result.get('finished', False):
                                vehicle_positions = result['positions']
                                tracks = result['tracks']
                                
                                # 将局部ID转换为全局ID
                                global_positions = []
                                for track_id, pos in vehicle_positions:
                                    global_id = reid_matcher.get_global_id(cam_id, track_id)
                                    if global_id != -1:
                                        global_positions.append((global_id, pos))
                                        
                                map_positions.extend(global_positions)
                                
                                # 绘制跟踪结果
                                frame_buffer[cam_id] = visualizer.draw_tracks(frame, tracks, cam_id, reid_matcher)
                    
                    # 更新地图
                    map_img = visualizer.update_map(map_positions)
                    
                    # 拼接所有帧和地图
                    tiled_img = visualizer.tile_frames(frame_buffer, map_img)
                    
                    # 显示结果
                    if args.display and tiled_img is not None:
                        cv2.imshow('Tunnel Vehicle Tracking', tiled_img)
                        key = cv2.waitKey(1)
                        if key == 27:  # Esc键退出
                            break
                    
                    # 如果需要保存视频
                    if args.save_video:
                        if output_video is None and tiled_img is not None:
                            h, w = tiled_img.shape[:2]
                            os.makedirs(args.output_dir, exist_ok=True)
                            output_path = os.path.join(args.output_dir, "combined_output.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output_video = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                        
                        if output_video is not None and tiled_img is not None:
                            output_video.write(tiled_img)
                    
                    # 清空帧缓冲区
                    frame_buffer = {}
            
            # 控制循环速度
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("用户中断，程序退出...")
    finally:
        # 释放资源
        if output_video is not None:
            output_video.release()
        cv2.destroyAllWindows()
        
        # 等待所有线程结束
        for thread in camera_threads:
            thread.join()

if __name__ == "__main__":
    main()
