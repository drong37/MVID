import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

class Visualizer:
    def __init__(self, map_size=(800, 600), camera_positions=None):
        """
        初始化可视化器
        
        参数:
            map_size: 地图尺寸 (宽, 高)
            camera_positions: 摄像头在地图上的位置 {camera_id: (x, y)}
        """
        self.map_size = map_size
        self.camera_positions = camera_positions or {}
        
        # 初始化地图
        self.map_img = np.ones((map_size[1], map_size[0], 3), dtype=np.uint8) * 255
        
        # 绘制摄像头位置
        for camera_id, pos in self.camera_positions.items():
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(self.map_img, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(self.map_img, f"Camera {camera_id}", (x-40, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                       
        # 车辆颜色映射
        self.color_map = {}
        self.next_color_idx = 0
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
        # 跟踪历史
        self.track_history = {}
        
    def draw_tracks(self, frame, tracks, camera_id, reid_matcher=None):
        """
        在帧上绘制跟踪
        
        参数:
            frame: 输入帧
            tracks: 跟踪结果，格式为 [x1, y1, x2, y2, track_id, class_id]
            camera_id: 摄像头ID
            reid_matcher: ReID匹配器 (可选)
            
        返回:
            绘制了跟踪的帧
        """
        frame_with_tracks = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取全局ID
            global_id = -1
            if reid_matcher is not None:
                global_id = reid_matcher.get_global_id(camera_id, track_id)
                
            # 如果没有全局ID，使用本地跟踪ID
            display_id = global_id if global_id != -1 else track_id
            
            # 获取颜色
            if display_id not in self.color_map:
                self.color_map[display_id] = self.colors[self.next_color_idx % len(self.colors)]
                self.next_color_idx += 1
                
            color = self.color_map[display_id]
            # 将matplotlib颜色转换为BGR
            color_rgb = tuple(int(c * 255) for c in mcolors.to_rgb(color))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # 绘制边界框
            cv2.rectangle(frame_with_tracks, (x1, y1), (x2, y2), color_bgr, 2)
            
            # 绘制ID
            id_text = f"ID: {display_id}"
            cv2.putText(frame_with_tracks, id_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
                       
            # 获取类别名称
            class_names = {2: "Car", 5: "Bus", 7: "Truck", 3: "Motorcycle"}
            class_name = class_names.get(int(class_id), "Vehicle")
            
            # 绘制类别
            cv2.putText(frame_with_tracks, class_name, (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
                       
        # 添加摄像头ID
        cv2.putText(frame_with_tracks, f"Camera {camera_id}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                   
        return frame_with_tracks
        
    def update_map(self, vehicles_info):
        """
        更新地图视图
        
        参数:
            vehicles_info: 车辆信息列表，格式为 [(global_id, (x, y))]
            
        返回:
            更新后的地图图像
        """
        # 复制干净的地图
        map_with_vehicles = self.map_img.copy()
        
        # 绘制车辆
        for global_id, pos in vehicles_info:
            x, y = int(pos[0]), int(pos[1])
            
            # 不在地图范围内的点不绘制
            if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
                continue
                
            # 获取颜色
            if global_id not in self.color_map:
                self.color_map[global_id] = self.colors[self.next_color_idx % len(self.colors)]
                self.next_color_idx += 1
                
            color = self.color_map[global_id]
            color_rgb = tuple(int(c * 255) for c in mcolors.to_rgb(color))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # 添加到跟踪历史
            if global_id not in self.track_history:
                self.track_history[global_id] = []
                
            self.track_history[global_id].append((x, y))
            
            # 只保留最近的50个点
            if len(self.track_history[global_id]) > 50:
                self.track_history[global_id] = self.track_history[global_id][-50:]
                
            # 绘制轨迹
            points = self.track_history[global_id]
            for i in range(1, len(points)):
                cv2.line(map_with_vehicles, points[i-1], points[i], color_bgr, 2)
                
            # 绘制当前位置
            cv2.circle(map_with_vehicles, (x, y), 8, color_bgr, -1)
            
            # 绘制ID
            cv2.putText(map_with_vehicles, f"ID: {global_id}", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
                       
        return map_with_vehicles
        
    def tile_frames(self, frames_dict, map_img=None):
        """
        将多个摄像头的帧和地图拼接到一个图像中
        
        参数:
            frames_dict: 帧字典 {camera_id: frame}
            map_img: 地图图像 (可选)
            
        返回:
            拼接的图像
        """
        if not frames_dict:
            return None
            
        # 获取每个帧的形状
        frames = list(frames_dict.values())
        h, w = frames[0].shape[:2]
        
        # 计算拼接布局
        n = len(frames)
        cols = min(n, 3) if map_img is not None else min(n, 4)
        rows = (n + cols - 1) // cols
        
        if map_img is not None:
            cols += 1
        
        # 创建拼接画布
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        # 放置每个帧
        for i, (camera_id, frame) in enumerate(frames_dict.items()):
            r, c = i // cols, i % cols
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = frame
            
        # 放置地图
        if map_img is not None:
            # 调整地图尺寸
            map_resized = cv2.resize(map_img, (w, h))
            # 放在最后一列
            canvas[0:h, (cols-1)*w:cols*w] = map_resized
            
        return canvas
