import cv2
import sys

from .rknn_func_yolo.rknn_pool import RKNNPoolExecutor
from .rknn_func_yolo.yolo_processor import YoloProcessor

from ..model import YoloModel


class YoloDetector:
    def __init__(self, yolo_model):
        if yolo_model == YoloModel.VEHICLE:
            model_path = sys.path[0] + "/sdk/data_layer/yolo/rknnModel/vehicle_quantized_mmse.rknn"
        elif yolo_model == YoloModel.WEAPON:
            model_path = sys.path[0] + "/sdk/data_layer/yolo/rknnModel/combat_quantized_mmse.rknn"
        else:
            raise RuntimeError(f"Unknown YoloModel type: {yolo_model}")

        yolo_processor = YoloProcessor(yolo_model)

        # 线程数, 增大可提高帧率
        self.worker_number = 4
        # 初始化 RKNN 池
        self.pool = RKNNPoolExecutor(
            rknn_model=model_path, worker_number=self.worker_number, func=yolo_processor.process)

        self.window_name = "Yolo Detect Image"

    def __del__(self):
        self.clean_up()

    def get_worker_number(self):
        return self.worker_number

    def fill_pool(self, frame):
        frame = cv2.resize(frame, (640, 480))
        self.pool.put(frame)

    def process_frame(self, frame):
        self.fill_pool(frame)
        (frame, detections), flag = self.pool.get()
        result = frame.copy()

        cv2.imshow(self.window_name, result)
        cv2.waitKey(1)

        return detections

    def clear_pool(self):
        frame_queue = self.pool.queue
        while not frame_queue.empty():
            frame_queue.get()

    def clean_up(self):
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        self.pool.release()
