import cv2

def convert_to_grayscale(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)
    # 将图片转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 保存灰度图
    cv2.imwrite(output_path, gray_image)
    return gray_image

if __name__ == "__main__":
    input_image_path = "./picture/frame_0.jpg"  # 替换为你的输入图片路径
    output_image_path = "./picture/gray_image_0.jpg"  # 替换为你想要保存的灰度图路径
    gray_img = convert_to_grayscale(input_image_path, output_image_path)
    # 显示灰度图
    cv2.imshow('Grayscale Image', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()