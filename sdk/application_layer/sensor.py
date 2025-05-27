from ..hardware_layer.manager.serial_manager import SerialManager
from ..hardware_layer.manager.camera_manager import CameraManager
from ..data_layer.grayscale import Grayscale
from .notice.base import NoticeBase
import cv2


class Sensor(NoticeBase):
    def __init__(self):
        super().__init__()

        # 硬件管理单例
        self.__serials = SerialManager()  # 串口
        self.__cameras = CameraManager()  # 相机

        # 传感器
        self.__grayscale = Grayscale(serial_port=self.__serials.get_serial_usb())  # 灰度

    def clean_up(self):
        print("释放摄像头和窗口资源...")
        if self.__cameras.get_cap_font().isOpened():
            self.__cameras.get_cap_font().release()
        cv2.destroyAllWindows()

    def get_grayscale(self):
        return self.__grayscale

    def get_camera(self):
        return self.__cameras.get_cap_font()
