import cv2
import numpy as np

def my_harris_corner_detector(image, k, kernel_size, window_size, threshold):
    """
    自定义 Harris 角点检测器

    参数:
    - image: 输入图像
    - k: Harris 检测器常数
    - kernel_size: 局部邻域核大小
    - window_size: NMS 窗口大小
    - threshold: 响应阈值

    返回:
    - corners: 检测到的角点坐标列表
    """
    # 1. 图像预处理
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    gray_image = np.float32(gray_image)

    # 2. 计算图像梯度
    Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # 3. 构建 H 矩阵
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    # 使用高斯窗口对梯度乘积进行求和
    s_x2 = cv2.GaussianBlur(Ix2, (kernel_size, kernel_size), 0)
    s_y2 = cv2.GaussianBlur(Iy2, (kernel_size, kernel_size), 0)
    s_xy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 0)

    # 4. 计算角点响应
    det_H = s_x2 * s_y2 - s_xy**2
    trace_H = s_x2 + s_y2
    R = det_H - k * (trace_H**2)

    # 5. 后处理
    # 阈值处理
    corners_img = R > threshold

    # 非极大值抑制 (NMS)
    h, w = corners_img.shape
    corners = []
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            window = R[y:y+window_size, x:x+window_size]
            if window.size == 0:
                continue
            max_val = window.max()
            if max_val > threshold:
                # 寻找最大值在窗口内的相对位置
                max_loc_rel = np.unravel_index(window.argmax(), window.shape)
                # 计算最大值在原图的绝对位置
                max_loc_abs = (y + max_loc_rel[0], x + max_loc_rel[1])
                
                # 确保该点是该窗口唯一的最大值
                if R[max_loc_abs] == max_val:
                    corners.append(max_loc_abs)

    # 去重
    corners = list(set(corners))
    
    # 将坐标转换为 (x, y) 格式
    corners = [(x, y) for y, x in corners]

    return corners
