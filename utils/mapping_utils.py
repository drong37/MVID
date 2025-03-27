import numpy as np
import json
import cv2

# 单应性矩阵映射
class CoordinateMapper:
    def __init__(self, camera_config_path):
        # 加载配置
        with open(camera_config_path, 'r') as f:
            self.camera_configs = json.load(f)
            
        # 为每个摄像头计算单应性矩阵
        self.homography_matrices = {}
        for camera_id, config in self.camera_configs.items():
            if 'image_points' in config and 'map_points' in config:
                src_pts = np.array(config['image_points'], dtype=np.float32)
                dst_pts = np.array(config['map_points'], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                self.homography_matrices[camera_id] = H
    
    def image_to_ground(self, camera_id, image_point, bbox_height=None):
        """
        将图像点映射到地图坐标
        """
        if camera_id not in self.homography_matrices:
            return None
            
        # 应用单应性变换
        H = self.homography_matrices[camera_id]
        p = np.array([[image_point[0], image_point[1], 1]], dtype=np.float32)
        p_transformed = cv2.perspectiveTransform(p.reshape(1, 1, 3), H)
        
        return np.array([p_transformed[0, 0, 0], p_transformed[0, 0, 1]])



# # 根据旋转矩阵和平移向量计算投影矩阵
# class CoordinateMapper:
#     def __init__(self, camera_config_path):
#         """
#         初始化坐标映射器
        
#         参数:
#             camera_config_path: 摄像头配置文件路径
#         """
#         # 加载摄像头配置
#         with open(camera_config_path, 'r') as f:
#             self.camera_configs = json.load(f)
            
#         # 初始化摄像头矩阵
#         self.camera_matrices = {}
#         for camera_id, config in self.camera_configs.items():
#             if camera_id in ['ground_normal', 'ground_point']:
#                 continue
                
#             # 摄像头内参矩阵
#             K = np.array(config['intrinsic_matrix']).reshape(3, 3)
#             # 摄像头外参矩阵 [R|t]
#             R = np.array(config['rotation_matrix']).reshape(3, 3)
#             t = np.array(config['translation_vector']).reshape(3, 1)
            
#             extrinsic = np.hstack((R, t))
            
#             # 摄像头投影矩阵 P = K[R|t]
#             P = K.dot(extrinsic)
            
#             self.camera_matrices[camera_id] = {
#                 'intrinsic': K,
#                 'extrinsic': extrinsic,
#                 'projection': P,
#                 'R': R,
#                 't': t
#             }
            
#         # 加载隧道地面平面参数（假设地面是平的）
#         self.ground_normal = np.array(self.camera_configs['ground_normal'])
#         self.ground_point = np.array(self.camera_configs['ground_point'])
            
#     def image_to_ground(self, camera_id, image_point, bbox_height):
#         """
#         将图像点转换为地面平面点
        
#         参数:
#             camera_id: 摄像头ID
#             image_point: 图像点 (x, y) - 边界框底部中心
#             bbox_height: 边界框高度（像素）
            
#         返回:
#             全局坐标系中的地面平面点 (x, y)
#         """
#         # 获取摄像头矩阵
#         camera_matrices = self.camera_matrices[camera_id]
#         K = camera_matrices['intrinsic']
#         R = camera_matrices['R']
#         t = camera_matrices['t'].squeeze()
        
#         # 将图像点转换为归一化坐标
#         image_point_homogeneous = np.array([image_point[0], image_point[1], 1.0])
#         ray_direction = np.linalg.inv(K).dot(image_point_homogeneous)
        
#         # 从摄像头中心到点的射线
#         camera_center = -np.linalg.inv(R).dot(t)
        
#         # 计算与地平面的交点
#         normal_dot_ray = np.dot(self.ground_normal, ray_direction)
        
#         # 如果射线与平面平行或接近平行，使用备用方法
#         if abs(normal_dot_ray) < 1e-6:
#             # 使用车辆高度作为估计依据
#             # 假设车辆平均高度为1.5米
#             vehicle_height = 1.5
#             # 使用边界框高度估计距离
#             # 焦距f = K[0,0]，实际高度H，图像高度h，距离D: h = f*H/D
#             focal_length = K[0, 0]
#             distance = vehicle_height * focal_length / bbox_height
            
#             # 计算地面点
#             direction = ray_direction / np.linalg.norm(ray_direction)
#             ground_point = camera_center + direction * distance
            
#             # 投影到地面
#             ground_point[1] = 0  # 假设y轴是向上的
#         else:
#             # 计算射线与地平面的交点
#             d = np.dot(self.ground_normal, (self.ground_point - camera_center)) / normal_dot_ray
#             ground_point = camera_center + d * ray_direction
        
#         # 只返回x和z坐标（假设y是高度）
#         return np.array([ground_point[0], ground_point[2]])
