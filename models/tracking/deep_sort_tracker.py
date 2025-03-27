import numpy as np
import torch
import sys
import os
import cv2

# 添加deep_sort包路径
sys.path.append('deep_sort_pytorch')
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

class VehicleTracker:
    def __init__(self, model_weights="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", 
                 max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100):
        """
        初始化DeepSORT车辆跟踪器
        
        参数:
            model_weights: DeepSORT模型权重路径
            max_dist: 特征距离阈值
            min_confidence: 检测框最小置信度
            max_iou_distance: 最大IOU距离
            max_age: 跟踪消失后保持的最大帧数
            n_init: 确认跟踪所需的连续检测次数
            nn_budget: 每个类别保持的特征数量
        """
        cfg = get_config()
        cfg.merge_from_dict({
            'DEEPSORT': {
                'REID_CKPT': model_weights,
                'MAX_DIST': max_dist,
                'MIN_CONFIDENCE': min_confidence,
                'MAX_IOU_DISTANCE': max_iou_distance,
                'MAX_AGE': max_age,
                'N_INIT': n_init,
                'NN_BUDGET': nn_budget
            }
        })
        
        self.tracker = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        
    def update(self, frame, detections):
        """
        使用新检测结果更新跟踪器
        
        参数:
            frame: 当前帧
            detections: 检测列表，格式为 [x1, y1, x2, y2, conf, class_id]
            
        返回:
            跟踪列表，格式为 [x1, y1, x2, y2, track_id, class_id]
        """
        if len(detections) == 0:
            return []
            
        bboxes = np.array([d[:4] for d in detections])
        scores = np.array([d[4] for d in detections])
        class_ids = np.array([d[5] for d in detections])
        
        # 将边界框格式从 [x1, y1, x2, y2] 转换为 [x1, y1, w, h]
        bboxes_xywh = np.zeros_like(bboxes)
        bboxes_xywh[:, 0] = bboxes[:, 0]
        bboxes_xywh[:, 1] = bboxes[:, 1]
        bboxes_xywh[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        bboxes_xywh[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        
        # 更新跟踪器
        outputs = self.tracker.update(bboxes_xywh, scores, class_ids, frame)
        
        # 将跟踪结果转换为 [x1, y1, x2, y2, track_id, class_id]
        results = []
        for output in outputs:
            x1, y1, w, h, track_id, class_id = output.tolist()
            x2, y2 = x1 + w, y1 + h
            results.append([x1, y1, x2, y2, track_id, class_id])
            
        return results
