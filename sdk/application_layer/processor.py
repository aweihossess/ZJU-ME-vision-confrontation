from .notice.base import NoticeBase
from ..logic_layer.line_follower import SingleLineFollower
from ..logic_layer.tag_detector import ApriltagDetector
from ..logic_layer.face_reid import FaceDetector
from ..logic_layer.yolo_detector import YoloDetector
from ..logic_layer.gesture_detector import GestureDetector


class Processor(NoticeBase):
    def __init__(self, yolo_model, show_image=False):
        super().__init__()
        self.yolo_model_enum = yolo_model

        self.__line_follower = SingleLineFollower()  # 巡线
        self.__apriltag_detector = ApriltagDetector(show=show_image)  # Apriltag 检测
        self.__face_detector = FaceDetector()  # 人脸识别
        self.__yolo_detector = YoloDetector(yolo_model, show=show_image)  # YOLO 识别
        self.__gesture_detector = GestureDetector()  # 手势识别

    def clean_up(self):
        print("释放视觉处理线程池资源...")
        self.__yolo_detector.clean_up()

    def get_line_follower(self):
        return self.__line_follower

    def get_apriltag_detector(self):
        return self.__apriltag_detector

    def get_face_detector(self):
        return self.__face_detector

    def get_yolo_detector(self):
        return self.__yolo_detector

    def get_gesture_detector(self):
        return self.__gesture_detector
