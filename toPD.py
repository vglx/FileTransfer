import numpy as np
import cv2
from open3d import geometry, io, utility

def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    """
    Convert a depth image into a point cloud.
    """
    height, width = depth_image.shape[:2]
    # 创建网格的x,y坐标
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # 将像素坐标转换为相机坐标
    Z = depth_image.astype(float)
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    
    # 堆叠成三维坐标
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    return points

def save_point_cloud(points, filename="point_cloud.ply"):
    """
    Save points to a PLY file.
    """
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(points)
    io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}.")

# 加载深度图像（这里只使用绿色通道）
depth_image_path = 'Figure_1.png'  # 你的深度图路径
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)[:, :, 1]  # 读取绿色通道

# 相机内参（这些值需要被替换为实际使用的值）
fx = 599.8959185504397
fy = 600.1453436994847
cx = 599.5
cy = 339.5

# 转换深度图到点云
points = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

# 保存点云到本地PLY文件
save_point_cloud(points, "output_point_cloud.ply")
