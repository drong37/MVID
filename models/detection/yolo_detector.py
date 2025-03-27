import torch
from ultralytics import YOLO
import numpy as np

class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt", device="cuda", conf_thres=0.25, iou_thres=0.45):
        """
        初始化YOLO车辆检测器
        
        参数:
            model_path: YOLO模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
            conf_thres: 检测置信度阈值
            iou_thres: NMS的IoU阈值
        """
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = YOLO(model_path)
        
        # COCO数据集中的车辆类别 (汽车, 卡车, 公交车, 摩托车)
        self.vehicle_classes = [2, 5, 7, 3]
        
    def detect(self, frame):
        """
        在一帧图像中检测车辆
        
        参数:
            frame: 输入帧 (numpy数组)
            
        返回:
            车辆边界框列表，格式为 [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]
        detections = []
        
        for result in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = result
            
            # 只保留车辆类别
            if int(class_id) in self.vehicle_classes:
                detections.append([x1, y1, x2, y2, conf, class_id])
                
        return detections
