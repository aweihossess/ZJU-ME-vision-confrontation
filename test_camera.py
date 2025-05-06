# 从 sdk.api 模块导入 UpAPI 类，该类可能封装了与相机及其他硬件交互的功能
from sdk.api import UpAPI
# 导入 OpenCV 库，用于处理和显示图像
import cv2

if __name__ == '__main__':
    # 创建 UpAPI 类的实例，用于后续调用相机相关功能
    api = UpAPI()

    # 进入无限循环，持续获取并显示相机画面
    while True:
        # 调用 api 的 get_camera_frame 方法，从相机获取一帧图像数据
        frame = api.get_camera_frame()
        # 使用 OpenCV 的 imshow 函数显示获取到的图像帧，窗口名称为 "camera"
        cv2.imshow("camera", frame)
        # 使用 OpenCV 的 waitKey 函数等待 1 毫秒，处理键盘事件，保持窗口响应
        cv2.waitKey(1)