import cv2
import numpy as np

def adjust_brightness(image, target_brightness=125, max_adjustment=80):
    """
    自适应亮度调整函数，根据图像当前亮度自动调整至目标亮度水平
    
    :param image: 输入图像
    :param target_brightness: 目标亮度值（0-255之间），默认为125
    :param max_adjustment: 最大调整幅度，防止过度调整，默认为80
    :return: 调整后的图像
    """        
    # 转换为灰度图计算平均亮度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 计算当前平均亮度
    current_brightness = np.mean(gray)
    
    # 计算需要调整的亮度值
    brightness_diff = target_brightness - current_brightness
    
    # 限制调整幅度
    if brightness_diff > max_adjustment:
        brightness_diff = max_adjustment
    elif brightness_diff < -max_adjustment:
        brightness_diff = -max_adjustment
        
    # 应用亮度调整
    adjusted_image = image.copy()
    if brightness_diff != 0:
        # 创建查找表进行亮度调整
        lut = np.arange(0, 256, dtype=np.uint8)
        
        # 应用亮度调整，确保值在0-255范围内
        for i in range(256):
            val = i + brightness_diff
            if val < 0:
                val = 0
            elif val > 255:
                val = 255
            lut[i] = val
            
        # 如果是彩色图像，对每个通道应用调整
        if len(image.shape) == 3:
            # 对每个通道应用查找表
            adjusted_image = cv2.LUT(image, lut)
        else:
            adjusted_image = cv2.LUT(image, lut)
    
    return adjusted_image

def auto_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    使用CLAHE（对比度受限的自适应直方图均衡化）增强图像对比度
    
    :param image: 输入图像
    :param clip_limit: 对比度限制参数，默认为2.0
    :param tile_grid_size: 分块大小，默认为8x8
    :return: 增强后的图像
    """
    if image is None:
        return None
        
    # 转换为LAB色彩空间（L通道代表亮度）
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 应用CLAHE到L通道
        cl = clahe.apply(l)
        
        # 合并通道
        enhanced_lab = cv2.merge((cl, a, b))
        
        # 转换回BGR色彩空间
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像直接应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)
    
    return enhanced_image 