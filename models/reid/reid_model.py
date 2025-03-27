import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os

# 添加原始ReID项目路径
sys.path.append(os.path.dirname(__file__))

from config import cfg
from model import make_model

class VehicleReID:
    def __init__(self, config_file, model_weights, device="cuda"):
        """
        初始化车辆ReID模型
        
        参数:
            config_file: ReID模型配置文件路径
            model_weights: ReID模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device
        
        # 加载配置
        cfg.merge_from_file(config_file)
        cfg.TEST.WEIGHT = model_weights
        cfg.freeze()
        
        # 创建模型
        self.model = make_model(cfg, num_class=1500)  # 类别数不重要，因为我们只使用特征提取
        self.model.load_param(cfg.TEST.WEIGHT)
        self.model.to(device)
        self.model.eval()
        
    def extract_features(self, image_patches):
        """
        从图像块中提取ReID特征
        
        参数:
            image_patches: 车辆图像块列表
            
        返回:
            归一化的特征向量
        """
        if not image_patches:
            return None
            
        with torch.no_grad():
            # 预处理图像块
            processed_patches = []
            for patch in image_patches:
                # 调整大小到模型输入尺寸
                patch = cv2.resize(patch, (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
                # 转换为张量
                patch = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0)
                # 归一化
                patch = patch.to(self.device) / 255.0
                mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1).to(self.device)
                std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1).to(self.device)
                patch = (patch - mean) / std
                processed_patches.append(patch)
                
            # 堆叠图像块
            patches_tensor = torch.cat(processed_patches, dim=0)
            
            # 提取特征
            features = self.model(patches_tensor)
            
            # 归一化特征
            if cfg.TEST.FEAT_NORM == 'yes':
                features = F.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy()
            
    def compute_distance(self, query_features, gallery_features):
        """
        计算查询特征和库特征之间的距离
        
        参数:
            query_features: 查询特征向量
            gallery_features: 库特征向量
            
        返回:
            距离矩阵
        """
        query_features = torch.from_numpy(query_features).to(self.device)
        gallery_features = torch.from_numpy(gallery_features).to(self.device)
        
        # 计算欧几里得距离
        m = query_features.size(0)
        n = gallery_features.size(0)
        dist_mat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
        
        return dist_mat.sqrt().cpu().numpy()
