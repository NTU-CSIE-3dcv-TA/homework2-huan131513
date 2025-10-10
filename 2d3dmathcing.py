from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    """
    計算描述符的平均值
    x: 形狀為 (N, descriptor_dim) 的numpy數組，包含多個描述符
    返回: 形狀為 (descriptor_dim,) 的列表，表示平均描述符
    數學公式: avg = (1/N) * Σ(x_i)，其中i從1到N
    """
    return list(np.mean(x,axis=0))  # axis=0表示沿著第0維（行）計算均值

def average_desc(train_df, points3D_df):
    """
    對每個3D點的多個描述符進行平均化處理
    train_df: 訓練數據DataFrame，包含POINT_ID, XYZ, RGB, DESCRIPTORS列
    points3D_df: 3D點數據DataFrame，包含POINT_ID和XYZ信息
    返回: 包含平均描述符的DataFrame
    """
    # 選擇需要的列：POINT_ID, XYZ, RGB, DESCRIPTORS
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]  # train_df形狀: (N, 4)
    
    # 按POINT_ID分組，將每個點的多個描述符堆疊成矩陣
    # desc: Series，每個元素是形狀為(M_i, descriptor_dim)的numpy數組
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)  # M_i是第i個點的描述符數量
    
    # 對每個點的描述符矩陣應用average函數，計算平均描述符
    # desc: Series，每個元素是形狀為(descriptor_dim,)的列表
    desc = desc.apply(average)  # 數學公式: avg_desc_i = (1/M_i) * Σ(desc_j)，j從1到M_i
    
    # 重置索引，將POINT_ID從索引轉為列
    desc = desc.reset_index()  # desc形狀: (num_points, 2)，列名為['POINT_ID', 'DESCRIPTORS']
    
    # 與points3D_df合併，添加XYZ坐標信息
    # 數學意義: 將平均描述符與3D點坐標關聯
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")  # 最終形狀: (num_points, 5)
    
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    """
    使用PnP算法求解相機姿態
    query: (kp_query, desc_query) - 查詢圖像的2D關鍵點和描述符
    model: (kp_model, desc_model) - 3D模型的3D點和對應描述符
    返回: (retval, rvec, tvec, inliers) - 求解結果
    數學原理: 求解方程 s*[u,v,1]^T = K*[R|t]*[X,Y,Z,1]^T
    """
    # 解包輸入參數
    kp_query, desc_query = query  # kp_query: (N, 2) 2D關鍵點坐標, desc_query: (N, descriptor_dim) 描述符
    kp_model, desc_model = model  # kp_model: (M, 3) 3D點坐標, desc_model: (M, descriptor_dim) 描述符
    
    # 相機內參矩陣 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])  # 形狀: (3, 3)
    
    # 畸變係數 [k1, k2, p1, p2, k3]
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])  # 形狀: (4,)

    # 使用BFMatcher進行描述符匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # L2距離，不使用交叉檢查
    matches = bf.knnMatch(desc_query, desc_model, k=2)  # 每個查詢描述符找2個最近鄰
    
    # 應用Lowe's比率測試過濾匹配
    # 數學條件: d1 < 0.75 * d2，其中d1是最小距離，d2是次小距離
    good_matches = []
    for match in matches:  # match: 包含2個DMatch對象的列表
        if len(match) == 2:  # 確保有2個匹配
            m, n = match  # m: 最近鄰匹配, n: 次近鄰匹配
            if m.distance < 0.75 * n.distance:  # Lowe's比率測試
                good_matches.append(m)  # 保留好的匹配
    
    # PnP算法至少需要4個對應點
    if len(good_matches) < 4:
        return None, None, None, None
    
    # 提取匹配的3D-2D對應點
    obj_pts = np.array([kp_model[m.trainIdx] for m in good_matches])  # 形狀: (n_matches, 3) 3D點
    img_pts = np.array([kp_query[m.queryIdx] for m in good_matches])  # 形狀: (n_matches, 2) 2D點
    
    # 使用RANSAC求解PnP問題
    # 數學目標: 最小化重投影誤差 Σ||p_i - π(K*[R|t]*P_i)||²
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, cameraMatrix, distCoeffs,  # 3D點, 2D點, 內參, 畸變
        iterationsCount=1000,    # RANSAC迭代次數
        reprojectionError=8.0,   # 重投影誤差閾值(像素)
        confidence=0.99          # 置信度
    )
    # rvec: 旋轉向量 (3,), tvec: 平移向量 (3,), inliers: 內點索引
    
    return retval, rvec, tvec, inliers

def rotation_error(R1, R2):
    """
    計算兩個四元數之間的旋轉誤差
    R1: 真實旋轉四元數，形狀 (1, 4) 或 (4,)，格式 [qx, qy, qz, qw]
    R2: 估計旋轉四元數，形狀 (1, 4) 或 (4,)，格式 [qx, qy, qz, qw]
    返回: 旋轉角度誤差（度）
    數學公式: θ = 2 * arccos(|q1·q2|)，其中q1·q2是四元數點積
    """
    # 將四元數轉換為Rotation對象
    rot1 = R.from_quat(R1.flatten())  # R1.flatten(): 形狀 (4,) 四元數 [qx, qy, qz, qw]
    rot2 = R.from_quat(R2.flatten())  # R2.flatten(): 形狀 (4,) 四元數 [qx, qy, qz, qw]
    
    # 計算相對旋轉: R_error = R_estimated * R_ground_truth^(-1)
    # 數學意義: 從真實姿態到估計姿態的旋轉
    rot_diff = rot2 * rot1.inv()  # rot1.inv()是R1的逆四元數
    
    # 獲取旋轉角度（軸角表示法）
    # 數學公式: θ = 2 * arccos(|w|)，其中w是四元數的標量部分
    angle = rot_diff.magnitude() * 180 / np.pi  # 轉換為度數

    return angle

def translation_error(t1, t2):
    """
    計算兩個平移向量之間的歐氏距離誤差
    t1: 真實平移向量，形狀 (1, 3) 或 (3,)，格式 [tx, ty, tz]
    t2: 估計平移向量，形狀 (1, 3) 或 (3,)，格式 [tx, ty, tz]
    返回: 平移誤差（米）
    數學公式: ||t1 - t2||₂ = √((tx1-tx2)² + (ty1-ty2)² + (tz1-tz2)²)
    """
    return np.linalg.norm(t1 - t2)  # 計算L2範數（歐氏距離）

def create_camera_pyramid(c2w_matrix, scale=0.1):
    """
    創建表示相機姿態的四角錐體
    c2w_matrix: 相機到世界坐標系的變換矩陣，形狀 (4, 4)
    scale: 錐體縮放因子，控制錐體大小
    返回: Open3D三角形網格對象
    數學意義: 在相機坐標系中定義錐體，然後變換到世界坐標系
    """
    try:
        import open3d as o3d
    except ImportError:
        return None
    
    # 在相機坐標系中定義錐體頂點
    # 頂點: 形狀 (5, 3)，包含1個頂點和4個底面頂點
    vertices = np.array([
        [0, 0, 0],              # 頂點 (光學中心) - 相機坐標系原點
        [-scale, -scale, scale], # 底面頂點1 - 左下
        [scale, -scale, scale],  # 底面頂點2 - 右下  
        [scale, scale, scale],   # 底面頂點3 - 右上
        [-scale, scale, scale]   # 底面頂點4 - 左上
    ])  # 數學意義: 錐體頂點在相機坐標系中的齊次坐標
    
    # 定義三角形面片
    # faces: 形狀 (6, 3)，每個面由3個頂點索引組成
    faces = np.array([
        [0, 1, 2],  # 面1: 頂點-左下-右下
        [0, 2, 3],  # 面2: 頂點-右下-右上
        [0, 3, 4],  # 面3: 頂點-右上-左上
        [0, 4, 1],  # 面4: 頂點-左上-左下
        [1, 2, 3],  # 面5: 底面三角形1 (左下-右下-右上)
        [1, 3, 4]   # 面6: 底面三角形2 (左下-右上-左上)
    ])  # 數學意義: 定義錐體的6個三角形面片
    
    # 創建Open3D三角形網格
    pyramid = o3d.geometry.TriangleMesh()
    pyramid.vertices = o3d.utility.Vector3dVector(vertices)  # 設置頂點
    pyramid.triangles = o3d.utility.Vector3iVector(faces)    # 設置面片
    
    # 設置頂點顏色 (RGB格式，範圍[0,1])
    colors = np.array([
        [1, 0, 0],  # 紅色 - 頂點 (光學中心)
        [0, 0, 1],  # 藍色 - 底面頂點1
        [0, 0, 1],  # 藍色 - 底面頂點2
        [0, 0, 1],  # 藍色 - 底面頂點3
        [0, 0, 1]   # 藍色 - 底面頂點4
    ])  # 形狀: (5, 3)，每個頂點對應一個RGB顏色
    pyramid.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # 將錐體從相機坐標系變換到世界坐標系
    # 數學公式: P_world = c2w_matrix * P_camera
    pyramid.transform(c2w_matrix)  # 應用4x4變換矩陣
    
    return pyramid

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    """
    可視化相機姿態和3D點雲
    Camera2World_Transform_Matrixs: 相機到世界變換矩陣列表，每個元素形狀 (4, 4)
    points3D_df: 3D點數據DataFrame，包含XYZ坐標和RGB顏色
    數學意義: 在3D空間中顯示重建的點雲和相機軌跡
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Skipping visualization.")
        print("Install with: pip install open3d")
        return
    
    # 從3D點數據創建點雲
    pcd = o3d.geometry.PointCloud()
    points = np.array(points3D_df["XYZ"].to_list())  # 形狀: (N, 3)，N個3D點
    pcd.points = o3d.utility.Vector3dVector(points)  # 設置點雲坐標
    
    # 添加顏色信息（如果可用）
    if "RGB" in points3D_df.columns:
        colors = np.array(points3D_df["RGB"].to_list()) / 255.0  # 形狀: (N, 3)，RGB值歸一化到[0,1]
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 設置點雲顏色
    else:
        # 默認灰色
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 統一灰色 (R=G=B=0.5)
    
    # 創建相機錐體和軌跡
    geometries = [pcd]  # 幾何對象列表，從點雲開始
    camera_positions = []  # 存儲相機位置，用於繪製軌跡
    
    # 為每個相機姿態創建四角錐體
    for i, c2w in enumerate(Camera2World_Transform_Matrixs):  # c2w形狀: (4, 4)
        # 創建表示相機姿態的四角錐體
        pyramid = create_camera_pyramid(c2w, scale=0.1)  # scale=0.1控制錐體大小
        if pyramid is not None:
            geometries.append(pyramid)  # 添加到幾何對象列表
        
        # 提取相機位置（變換矩陣的平移部分）
        camera_positions.append(c2w[:3, 3])  # 形狀: (3,)，相機在世界坐標系中的位置
    
    # 創建相機軌跡線
    if len(camera_positions) > 1:  # 至少需要2個相機位置才能繪製軌跡
        trajectory_points = np.array(camera_positions)  # 形狀: (N_cameras, 3)
        # 創建連接相鄰相機的線段
        trajectory_lines = [[i, i+1] for i in range(len(camera_positions)-1)]  # 形狀: (N_cameras-1, 2)
        
        # 創建線段集合
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)  # 設置線段端點
        line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)    # 設置線段連接
        # 設置線段顏色為綠色
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(trajectory_lines))])
        
        geometries.append(line_set)  # 添加軌跡線到幾何對象列表
    
    # 顯示3D可視化
    o3d.visualization.draw_geometries(geometries,
                                     window_name="Camera Poses and 3D Points",
                                     width=1024, height=768)  # 窗口大小

def create_virtual_cube_points(size=1.0, resolution=20):
    """
    創建虛擬立方體點雲，每個面有不同的顏色
    size: 立方體大小
    resolution: 每個面的點密度
    返回: (points, colors) - 3D點坐標和對應顏色
    """
    points = []
    colors = []
    
    # 六個面的顏色 (RGB格式)
    face_colors = [
        [1, 0, 0],    # 紅色 - 右面
        [0, 1, 0],    # 綠色 - 左面  
        [0, 0, 1],    # 藍色 - 頂面
        [1, 1, 0],    # 黃色 - 底面
        [1, 0, 1],    # 洋紅色 - 前面
        [0, 1, 1],    # 青色 - 後面
    ]
    
    # 為每個面生成點
    for i in range(resolution):
        for j in range(resolution):
            u = i / (resolution - 1)
            v = j / (resolution - 1)
            
            # 右面 (x = size/2)
            points.append([size/2, (u-0.5)*size, (v-0.5)*size])
            colors.append(face_colors[0])
            
            # 左面 (x = -size/2)
            points.append([-size/2, (u-0.5)*size, (v-0.5)*size])
            colors.append(face_colors[1])
            
            # 頂面 (y = size/2)
            points.append([(u-0.5)*size, size/2, (v-0.5)*size])
            colors.append(face_colors[2])
            
            # 底面 (y = -size/2)
            points.append([(u-0.5)*size, -size/2, (v-0.5)*size])
            colors.append(face_colors[3])
            
            # 前面 (z = size/2)
            points.append([(u-0.5)*size, (v-0.5)*size, size/2])
            colors.append(face_colors[4])
            
            # 後面 (z = -size/2)
            points.append([(u-0.5)*size, (v-0.5)*size, -size/2])
            colors.append(face_colors[5])
    
    return np.array(points), np.array(colors)

def painters_algorithm_sort(points_3d, camera_pose):
    """
    畫家算法：按深度排序體素（從最遠到最近）
    points_3d: 3D點坐標，形狀 (N, 3)
    camera_pose: 相機姿態矩陣，形狀 (4, 4)
    返回: 排序後的索引
    """
    # 將3D點轉換到相機坐標系
    points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_camera = (camera_pose @ points_homogeneous.T).T
    
    # 獲取深度（相機坐標系中的Z坐標）
    depths = points_camera[:, 2]
    
    # 過濾無效深度值
    valid_depth_mask = np.isfinite(depths) & (depths > 0)
    
    if not np.any(valid_depth_mask):
        return np.array([])
    
    # 按深度排序（最遠的在前）
    valid_depths = depths[valid_depth_mask]
    sorted_indices = np.argsort(valid_depths)[::-1]  # 反向排序，最遠的在前
    
    # 映射回原始索引
    valid_indices = np.where(valid_depth_mask)[0]
    return valid_indices[sorted_indices]

def project_points_to_image(points_3d, camera_matrix, camera_pose, image_size):
    """
    將3D點投影到2D圖像坐標
    points_3d: 3D點坐標，形狀 (N, 3)
    camera_matrix: 相機內參矩陣，形狀 (3, 3)
    camera_pose: 相機姿態矩陣，形狀 (4, 4)
    image_size: 圖像尺寸 (height, width)
    返回: (u, v, valid_mask) - 像素坐標和有效掩碼
    """
    # 轉換到相機坐標系
    points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # 檢查輸入數據的有效性
    if not np.isfinite(points_homogeneous).all():
        print("警告：輸入3D點包含無效值")
        return np.array([]), np.array([]), np.array([])
    
    if not np.isfinite(camera_pose).all():
        print("警告：相機姿態矩陣包含無效值")
        return np.array([]), np.array([]), np.array([])
    
    points_camera = (camera_pose @ points_homogeneous.T).T
    
    # 過濾相機前方的點和無效點
    valid_mask = (points_camera[:, 2] > 0) & np.isfinite(points_camera).all(axis=1)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    points_camera = points_camera[valid_mask]
    
    # 投影到圖像平面
    points_2d = points_camera[:, :2] / points_camera[:, 2:3]
    
    # 檢查投影後的有效性
    if not np.isfinite(points_2d).all():
        print("警告：投影後的2D點包含無效值")
        return np.array([]), np.array([]), np.array([])
    
    points_2d_homogeneous = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
    points_2d_projected = (camera_matrix @ points_2d_homogeneous.T).T
    
    # 轉換為像素坐標
    u = points_2d_projected[:, 0].astype(int)
    v = points_2d_projected[:, 1].astype(int)
    
    # 過濾圖像邊界內的點
    valid_pixels = (u >= 0) & (u < image_size[1]) & (v >= 0) & (v < image_size[0])
    
    if not np.any(valid_pixels):
        return np.array([]), np.array([]), np.array([])
    
    return u[valid_pixels], v[valid_pixels], valid_mask[valid_pixels]

def render_virtual_cube_on_image(image, cube_points, cube_colors, camera_matrix, camera_pose):
    """
    在圖像上渲染虛擬立方體，使用畫家算法
    image: 輸入圖像
    cube_points: 立方體3D點，形狀 (N, 3)
    cube_colors: 立方體顏色，形狀 (N, 3)
    camera_matrix: 相機內參矩陣
    camera_pose: 相機姿態矩陣
    返回: 渲染後的圖像
    """
    image_with_cube = image.copy()
    height, width = image.shape[:2]
    
    # 投影點到圖像
    u, v, valid_mask = project_points_to_image(cube_points, camera_matrix, camera_pose, (height, width))
    
    if len(u) == 0:
        return image_with_cube
    
    # 獲取有效點和顏色
    valid_points = cube_points[valid_mask]
    valid_colors = cube_colors[valid_mask]
    
    # 使用畫家算法排序
    sorted_indices = painters_algorithm_sort(valid_points, camera_pose)
    
    if len(sorted_indices) == 0:
        return image_with_cube
    
    # 按深度順序排序
    sorted_colors = valid_colors[sorted_indices]
    sorted_u = u[sorted_indices]
    sorted_v = v[sorted_indices]
    
    # 渲染點（從最遠到最近）
    for i in range(len(sorted_u)):
        if 0 <= sorted_u[i] < width and 0 <= sorted_v[i] < height:
            # 轉換為BGR格式
            color_bgr = (int(sorted_colors[i][2] * 255), 
                        int(sorted_colors[i][1] * 255), 
                        int(sorted_colors[i][0] * 255))
            # 繪製較大的點以便可見
            cv2.circle(image_with_cube, (sorted_u[i], sorted_v[i]), 3, color_bgr, -1)
            # 添加白色邊框增加對比度
            cv2.circle(image_with_cube, (sorted_u[i], sorted_v[i]), 4, (255, 255, 255), 1)
    
    return image_with_cube

def project_point_cloud_to_image(points_3d, colors_3d, camera_matrix, camera_pose, image_size):
    """
    將3D點雲投影到2D圖像，返回投影後的點和顏色
    points_3d: 3D點坐標，形狀 (N, 3)
    colors_3d: 3D點顏色，形狀 (N, 3)
    camera_matrix: 相機內參矩陣，形狀 (3, 3)
    camera_pose: 相機姿態矩陣，形狀 (4, 4)
    image_size: 圖像尺寸 (height, width)
    返回: (u, v, colors, depths) - 像素坐標、顏色和深度
    """
    # 轉換到相機坐標系
    points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_camera = (camera_pose @ points_homogeneous.T).T
    
    # 過濾相機前方的點和無效點
    valid_mask = (points_camera[:, 2] > 0) & np.isfinite(points_camera).all(axis=1)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    points_camera = points_camera[valid_mask]
    colors_valid = colors_3d[valid_mask]
    
    # 投影到圖像平面
    points_2d = points_camera[:, :2] / points_camera[:, 2:3]
    points_2d_homogeneous = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
    points_2d_projected = (camera_matrix @ points_2d_homogeneous.T).T
    
    # 轉換為像素坐標
    u = points_2d_projected[:, 0].astype(int)
    v = points_2d_projected[:, 1].astype(int)
    
    # 過濾圖像邊界內的點
    valid_pixels = (u >= 0) & (u < image_size[1]) & (v >= 0) & (v < image_size[0])
    
    if not np.any(valid_pixels):
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return u[valid_pixels], v[valid_pixels], colors_valid[valid_pixels], points_camera[valid_pixels, 2]

def render_ar_scene(image, points3D_df, cube_points, cube_colors, camera_matrix, camera_pose):
    """
    渲染AR場景：只渲染虛擬立方體（不顯示點雲背景）
    image: 輸入圖像
    points3D_df: 3D點雲數據（不使用，保留參數以兼容）
    cube_points: 立方體3D點
    cube_colors: 立方體顏色
    camera_matrix: 相機內參矩陣
    camera_pose: 相機姿態矩陣
    返回: 渲染後的圖像
    """
    height, width = image.shape[:2]
    image_with_ar = image.copy()
    
    # 只投影立方體到圖像（不顯示點雲）
    cube_u, cube_v, cube_colors_proj, cube_depths = project_point_cloud_to_image(
        cube_points, cube_colors, camera_matrix, camera_pose, (height, width)
    )
    
    if len(cube_u) == 0:
        return image_with_ar
    
    # 畫家算法：按深度排序（最遠的在前）
    depth_sorted_indices = np.argsort(cube_depths)[::-1]
    
    # 按深度順序渲染立方體點
    for i in depth_sorted_indices:
        if 0 <= cube_u[i] < width and 0 <= cube_v[i] < height:
            # 轉換為BGR格式
            color_bgr = (int(cube_colors_proj[i][2] * 255), 
                        int(cube_colors_proj[i][1] * 255), 
                        int(cube_colors_proj[i][0] * 255))
            
            # 繪製點（稍微大一點以便看清）
            cv2.circle(image_with_ar, (cube_u[i], cube_v[i]), 3, color_bgr, -1)
            # 添加白色邊框增加對比度
            cv2.circle(image_with_ar, (cube_u[i], cube_v[i]), 4, (255, 255, 255), 1)
    
    return image_with_ar

def create_ar_video(images_df, points3D_df, train_df, point_desc_df, r_list, t_list):
    """
    創建增強現實環場影片：在原始圖像上疊加虛擬立方體
    images_df: 圖像信息
    points3D_df: 3D點雲數據（用於相機姿態）
    train_df: 訓練數據
    point_desc_df: 點描述符數據
    r_list: 旋轉四元數列表
    t_list: 平移向量列表
    """
    print("\n=== Problem 2-2: 創建增強現實環場影片 ===")
    
    # 相機內參矩陣
    camera_matrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    
    # 創建虛擬立方體
    cube_points, cube_colors = create_virtual_cube_points(size=0.1, resolution=15)
    
    # 定義立方體變換（位置、方向、縮放）
    # 將立方體放在相機前方，更容易看到
    cube_rotation = np.array([0, 0, 0])  # 度數 - 不旋轉
    cube_translation = np.array([0.0, 0.0, 3.0])  # 米 - 放在相機前方3米
    cube_scale = 3.0  # 放大立方體
    
    # 應用變換到立方體
    from transform_cube import get_transform_mat
    cube_transform_3x4 = get_transform_mat(cube_rotation, cube_translation, cube_scale)
    
    # 將3x4變換矩陣轉換為4x4齊次變換矩陣
    cube_transform = np.eye(4)
    cube_transform[:3, :] = cube_transform_3x4
    
    # 檢查變換矩陣的有效性
    if not np.isfinite(cube_transform).all():
        print("警告：立方體變換矩陣包含無效值，使用默認變換")
        cube_transform = np.eye(4)
        cube_transform[:3, 3] = cube_translation
    
    # 直接使用簡單的變換，避免數值問題
    cube_points_transformed = cube_points * cube_scale + cube_translation
    
    print(f"立方體變換前形狀: {cube_points.shape}")
    print(f"立方體變換後形狀: {cube_points_transformed.shape}")
    print(f"立方體變換後範圍: X[{cube_points_transformed[:, 0].min():.2f}, {cube_points_transformed[:, 0].max():.2f}], Y[{cube_points_transformed[:, 1].min():.2f}, {cube_points_transformed[:, 1].max():.2f}], Z[{cube_points_transformed[:, 2].min():.2f}, {cube_points_transformed[:, 2].max():.2f}]")
    
    # 處理所有驗證圖像 - 使用所有可用的圖像
    # 獲取所有圖像ID並按照圖像名稱中的編號排序
    # 從圖像名稱中提取編號 (例如: train_img64.jpg -> 64)
    def extract_img_number(row):
        name = row["NAME"]
        # 提取數字部分：train_img64.jpg -> 64
        import re
        match = re.search(r'img(\d+)', name)
        if match:
            return int(match.group(1))
        return 0
    
    # 添加一個臨時列來存儲圖像編號
    images_df_sorted = images_df.copy()
    images_df_sorted['img_number'] = images_df_sorted.apply(extract_img_number, axis=1)
    
    # 按圖像編號排序
    images_df_sorted = images_df_sorted.sort_values('img_number')
    
    # 獲取排序後的圖像ID列表
    IMAGE_ID_LIST = images_df_sorted["IMAGE_ID"].tolist()
    print(f"總共將處理 {len(IMAGE_ID_LIST)} 張圖像（按圖像編號排序）")
    print(f"第一張圖像: {images_df_sorted.iloc[0]['NAME']}, 最後一張圖像: {images_df_sorted.iloc[-1]['NAME']}")
    
    # 創建輸出資料夾
    import os
    ar_frames_dir = "ar_frames"
    if not os.path.exists(ar_frames_dir):
        os.makedirs(ar_frames_dir)
        print(f"已創建資料夾: {ar_frames_dir}/")
    
    output_images = []
    
    print("正在創建AR環場影片...")
    for i, idx in enumerate(tqdm(IMAGE_ID_LIST)):
        # 加載圖像
        fname = images_df.loc[images_df["IMAGE_ID"] == idx]["NAME"].values[0]
        image = cv2.imread("data/frames/" + fname)
        
        if image is None:
            print(f"無法加載圖像: {fname}")
            continue
        
        # 使用估計的相機姿態
        if i < len(r_list) and i < len(t_list):
            # 將四元數轉換為旋轉矩陣
            rot = R.from_quat(r_list[i].flatten())
            R_mat = rot.as_matrix()
            
            # 檢查旋轉矩陣的有效性
            if not np.isfinite(R_mat).all():
                print(f"警告：圖像 {idx} 的旋轉矩陣包含無效值，跳過")
                continue
            
            # 創建相機姿態矩陣（世界到相機）
            # 注意：PnP返回的是相機在世界坐標系中的姿態
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R_mat.T  # 轉置得到世界到相機的旋轉
            camera_pose[:3, 3] = -R_mat.T @ t_list[i].flatten()  # 計算世界到相機的平移
            
            # 檢查最終相機姿態矩陣的有效性
            if not np.isfinite(camera_pose).all():
                print(f"警告：圖像 {idx} 的相機姿態矩陣包含無效值，跳過")
                continue
            
        else:
            # 如果沒有對應的相機姿態，使用真值
            print(f"使用真值相機姿態 for 圖像 {idx}")
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            if len(ground_truth) > 0:
                rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values[0]
                tvec_gt = ground_truth[["TX","TY","TZ"]].values[0]
                
                # 將四元數轉換為旋轉矩陣
                rot = R.from_quat(rotq_gt)
                R_mat = rot.as_matrix()
                
                # 創建相機姿態矩陣（世界到相機）
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R_mat.T
                camera_pose[:3, 3] = -R_mat.T @ tvec_gt
            else:
                print(f"跳過圖像 {idx}：沒有相機姿態數據")
                continue
        
        # 渲染AR場景：點雲背景 + 虛擬立方體
        image_with_ar = render_ar_scene(
            image, points3D_df, cube_points_transformed, cube_colors, 
            camera_matrix, camera_pose
        )
        
        # 添加調試信息
        cv2.putText(image_with_ar, f"AR Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if i < len(t_list):
            cv2.putText(image_with_ar, f"Camera: [{t_list[i][0][0]:.2f}, {t_list[i][0][1]:.2f}, {t_list[i][0][2]:.2f}]", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(image_with_ar, f"Camera: Ground Truth", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_with_ar, "AR Virtual Cube", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_images.append(image_with_ar)
        
        # 每10幀保存一個示例到ar_frames資料夾
        if i % 10 == 0:
            output_path = os.path.join(ar_frames_dir, f"ar_frame_{idx:03d}.jpg")
            cv2.imwrite(output_path, image_with_ar)
    
    # 創建影片
    if len(output_images) > 0:
        height, width = output_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 設置合適的幀率，讓影片播放流暢
        fps = 10.0  # 每秒10幀
        video_writer = cv2.VideoWriter('ar_video.mp4', fourcc, fps, (width, height))
        
        # 直接寫入所有幀
        for img in output_images:
            video_writer.write(img)
        
        video_writer.release()
        print(f"AR環場影片已保存為: ar_video.mp4 (共{len(output_images)}幀，FPS={fps})")
    
    print("=== 增強現實環場影片創建完成 ===")
    return output_images

if __name__ == "__main__":
    """
    主程序：2D-3D匹配和相機姿態估計
    流程：加載數據 → 處理描述符 → 對每張圖像進行PnP求解 → 計算誤差 → 可視化
    """
    # 加載數據文件
    images_df = pd.read_pickle("data/images.pkl")      # 圖像信息，包含相機姿態真值
    train_df = pd.read_pickle("data/train.pkl")        # 訓練數據，包含2D-3D對應關係
    points3D_df = pd.read_pickle("data/points3D.pkl")  # 3D點雲數據
    point_desc_df = pd.read_pickle("data/point_desc.pkl")  # 每張圖像的2D關鍵點和描述符

    # 處理模型描述符：對每個3D點的多個描述符進行平均化
    desc_df = average_desc(train_df, points3D_df)  # 返回包含平均描述符的DataFrame
    kp_model = np.array(desc_df["XYZ"].to_list())  # 形狀: (N, 3)，3D點坐標
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)  # 形狀: (N, descriptor_dim)，平均描述符

    # 定義要處理的驗證圖像ID列表
    IMAGE_ID_LIST = [200, 201]  # 驗證圖像ID
    r_list = []                 # 存儲估計的旋轉四元數
    t_list = []                 # 存儲估計的平移向量
    rotation_error_list = []    # 存儲旋轉誤差
    translation_error_list = [] # 存儲平移誤差
    # 對每張驗證圖像進行相機姿態估計
    for idx in tqdm(IMAGE_ID_LIST):  # 使用進度條顯示處理進度
        # 加載查詢圖像
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]  # 獲取圖像文件名
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)  # 讀取灰度圖像，形狀: (H, W)

        # 加載查詢圖像的2D關鍵點和描述符
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]  # 篩選當前圖像的數據
        kp_query = np.array(points["XY"].to_list())  # 形狀: (M, 2)，2D關鍵點坐標
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)  # 形狀: (M, descriptor_dim)，描述符

        # 使用PnP算法求解相機姿態
        # 數學目標: 求解 s*[u,v,1]^T = K*[R|t]*[X,Y,Z,1]^T
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        # retval: 求解是否成功, rvec: 旋轉向量 (3,), tvec: 平移向量 (3,), inliers: 內點索引
        
        if rvec is None:  # PnP求解失敗
            print(f"PnP failed for image {idx}")
            continue
            
        # 將旋轉向量轉換為四元數表示
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()  # 形狀: (1, 4)，四元數 [qx, qy, qz, qw]
        tvec = tvec.reshape(1,3)  # 形狀: (1, 3)，平移向量 [tx, ty, tz]
        r_list.append(rotq)  # 存儲估計的旋轉
        t_list.append(tvec)  # 存儲估計的平移

        # 獲取相機姿態真值（用於誤差計算）
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]  # 篩選當前圖像的真值
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values  # 形狀: (1, 4)，真實四元數
        tvec_gt = ground_truth[["TX","TY","TZ"]].values  # 形狀: (1, 3)，真實平移向量

        # 計算姿態估計誤差
        r_error = rotation_error(rotq_gt, rotq)  # 旋轉誤差（度）
        t_error = translation_error(tvec_gt, tvec)  # 平移誤差（米）
        rotation_error_list.append(r_error)  # 存儲旋轉誤差
        translation_error_list.append(t_error)  # 存儲平移誤差

    # 計算中位數誤差統計
    if rotation_error_list and translation_error_list:  # 確保有有效的誤差數據
        # 計算中位數誤差（更魯棒的統計量，不受異常值影響）
        median_rotation_error = np.median(rotation_error_list)  # 旋轉誤差中位數（度）
        median_translation_error = np.median(translation_error_list)  # 平移誤差中位數（米）
        
        print(f"Median Rotation Error: {median_rotation_error:.4f} degrees")
        print(f"Median Translation Error: {median_translation_error:.4f}")
        
        # 輸出詳細的誤差統計信息
        print(f"Rotation Error Statistics:")
        print(f"  - Mean: {np.mean(rotation_error_list):.4f} degrees")  # 平均旋轉誤差
        print(f"  - Std: {np.std(rotation_error_list):.4f} degrees")    # 旋轉誤差標準差
        print(f"  - Min: {np.min(rotation_error_list):.4f} degrees")    # 最小旋轉誤差
        print(f"  - Max: {np.max(rotation_error_list):.4f} degrees")    # 最大旋轉誤差
        
        print(f"Translation Error Statistics:")
        print(f"  - Mean: {np.mean(translation_error_list):.4f}")       # 平均平移誤差
        print(f"  - Std: {np.std(translation_error_list):.4f}")         # 平移誤差標準差
        print(f"  - Min: {np.min(translation_error_list):.4f}")         # 最小平移誤差
        print(f"  - Max: {np.max(translation_error_list):.4f}")         # 最大平移誤差
    else:
        print("No valid pose estimates obtained.")

    # 結果可視化：將相機姿態轉換為相機到世界的變換矩陣
    Camera2World_Transform_Matrixs = []  # 存儲所有相機的c2w變換矩陣
    for r, t in zip(r_list, t_list):  # 遍歷每對估計的旋轉和平移
        # 將四元數轉換為旋轉矩陣
        # r: 形狀 (1, 4)，四元數 [qx, qy, qz, qw]
        rot = R.from_quat(r.flatten())  # 轉換為Rotation對象
        R_mat = rot.as_matrix()  # 形狀: (3, 3)，旋轉矩陣
        
        # 創建相機到世界的變換矩陣
        # 數學公式: c2w = [[R^T, -R^T*t], [0, 1]]
        c2w = np.eye(4)  # 形狀: (4, 4)，單位矩陣
        c2w[:3, :3] = R_mat.T  # 旋轉部分：R^T（轉置）
        c2w[:3, 3] = -R_mat.T @ t.flatten()  # 平移部分：-R^T*t
        
        Camera2World_Transform_Matrixs.append(c2w)  # 添加到變換矩陣列表
    
    # 調用可視化函數顯示結果
    visualization(Camera2World_Transform_Matrixs, points3D_df)
    
    # Problem 2-2: 創建增強現實影片
    create_ar_video(images_df, points3D_df, train_df, point_desc_df, r_list, t_list)