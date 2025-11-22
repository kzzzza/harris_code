import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from harris_detector import my_harris_corner_detector
import os

def main():
    parser = argparse.ArgumentParser(description='Custom Harris Corner Detector')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--k', type=float, default=0.04, help='Harris detector constant.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Neighborhood kernel size for gradient calculation.')
    parser.add_argument('--window_size', type=int, default=10, help='NMS window size.')
    parser.add_argument('--threshold', type=float, default=10000, help='Response threshold.')
    args = parser.parse_args()

    # 读取图像
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # --- 1. 在原始图像上运行自定义检测器 ---
    print("Running custom Harris detector on the original image...")
    corners_custom = my_harris_corner_detector(image, args.k, args.kernel_size, args.window_size, args.threshold)
    
    image_with_corners_custom = image.copy()
    for x, y in corners_custom:
        image_with_corners_custom[y, x] = [0, 0, 255]

    # --- 2. 旋转鲁棒性测试 ---
    print("\n--- Rotation Robustness Test ---")
    # 旋转图像45度
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    print("Running custom Harris detector on the rotated image...")
    corners_rotated = my_harris_corner_detector(rotated_image, args.k, args.kernel_size, args.window_size, args.threshold)

    rotated_image_with_corners = rotated_image.copy()
    for x, y in corners_rotated:
        rotated_image_with_corners[y, x] = [0, 0, 255]

    # --- 3. 与 OpenCV 比较 ---
    print("\n--- Comparison with OpenCV ---")
    print("Running cv2.cornerHarris...")
    gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_cv = np.float32(gray_cv)
    # OpenCV的cornerHarris参数与我们的稍有不同，block_size对应我们的kernel_size
    dst_cv = cv2.cornerHarris(gray_cv, blockSize=args.kernel_size, ksize=3, k=args.k)
    
    # OpenCV的结果需要进行阈值处理来标记角点
    image_with_corners_cv = image.copy()
    # 我们使用一个相对阈值，这在不同图像间更鲁棒
    image_with_corners_cv[dst_cv > 0.01 * dst_cv.max()] = [0, 0, 255]


    # --- 4. 可视化与报告 ---
    plt.figure(figsize=(18, 12))

    # 自定义检测器结果
    plt.subplot(2, 2, 1)
    plt.title('Custom Detector on Original Image')
    plt.imshow(cv2.cvtColor(image_with_corners_custom, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 旋转测试结果
    plt.subplot(2, 2, 2)
    plt.title('Custom Detector on Rotated Image (45 deg)')
    plt.imshow(cv2.cvtColor(rotated_image_with_corners, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # OpenCV 比较结果
    plt.subplot(2, 2, 3)
    plt.title('OpenCV cv2.cornerHarris Result')
    plt.imshow(cv2.cvtColor(image_with_corners_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 文本报告：打印并显示使用的命令行参数
    report_text = f"""Used command-line arguments:
    image_path: {args.image_path}
    k: {args.k}
    kernel_size: {args.kernel_size}
    window_size: {args.window_size}
    threshold: {args.threshold}
    """
    print(report_text)
    
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.0, 0.95, report_text, va='top', fontsize=10, wrap=True)


    plt.tight_layout()
    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    plt.savefig(f'harris_analysis_results_of_{img_name}.png')
    print(f"\nResults saved to harris_analysis_results_of_{img_name}.png")
    plt.show()


if __name__ == '__main__':
    main()
