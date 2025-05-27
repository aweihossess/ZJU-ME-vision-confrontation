# 从 sdk.api 模块导入 UpAPI 类，该类可能封装了与相机及其他硬件交互的功能
from sdk.api import UpAPI
# 导入 OpenCV 库，用于处理和显示图像
import cv2
from sdk.data_layer.arm import arm_action_factory as arm_data

if __name__ == '__main__':
    api = UpAPI()
    import os
    save_folder = os.path.expanduser('~/code_vision/app/picture')  # 指定保存图片的文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    frame_count = 0

    arm_reset = {
        "left": arm_data.left_arm_clamp(),
        "right": arm_data.right_arm_clamp()
    } # 复位动作



    while True:
        api.execute_arm_action(arm_reset["left"], arm_reset["right"])
    # 执行并且保持复位动作
        frame = api.get_camera_frame()
        cv2.imshow("camera", frame)
        key = cv2.waitKey(1)
        if key == ord('k'):
            image_path = os.path.join(save_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(image_path, frame)
            print(f'图像已保存到 {image_path}')
            frame_count += 1
