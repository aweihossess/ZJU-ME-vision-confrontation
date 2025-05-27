import cv2
import numpy as np

def extract_black_logo(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 将图片从 BGR 转换为 Lab 颜色空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 定义 Lab 颜色空间中黑色的范围
    lower_black = np.array([0, 120, 120])  # 调整黑色范围
    upper_black = np.array([50, 136, 136])

    # 创建掩码，提取黑色区域
    mask = cv2.inRange(lab_image, lower_black, upper_black)

    # 形态学操作：膨胀和腐蚀（可选，用于优化掩码）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 设定面积阈值，用于筛选大块黑色区域
    area_threshold = 1000  # 可根据实际情况调整

   # 创建一张全白的掩码
    new_mask = np.ones_like(mask) * 255

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            # 填充大块黑色区域为白色
            cv2.drawContours(new_mask, [contour], -1, 0, 10)

    # 创建一张与原图大小相同的白色图片
    result = np.ones_like(image) * 255  # 白色背景

    # 将掩码对应的黑色区域复制到新图片上
    result[mask == 0] = image[mask == 0]

    return result


if __name__ == "__main__": 
    for i in range(9):
        # print(f"开始提取 frame_{i} 的黑色logo") 
        # image_path = f"./picture_used/frame_{i}.jpg"
        # save_path = f"./picture_used/Lab_{i}.jpg"
        print(f"开始提取 frame_0 的黑色logo") 
        image_path = f"./picture_used/frame_0.jpg"
        save_path = f"./picture_used/Lab_0.jpg"
        black_logo = extract_black_logo(image_path) 
        print(f"提取 frame_{i} 的黑色logo完成") 
        # 保存结果 
        cv2.imwrite(save_path, black_logo) 
    
        # 显示结果（如果不需要逐个显示，可以注释掉下面三行）
        cv2.imshow('Extracted Black Logo', black_logo) 
        cv2.waitKey(3000) 
        cv2.destroyAllWindows()

