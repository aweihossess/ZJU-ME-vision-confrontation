# 库导入
from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.pid import PIDController
from sdk.logic_layer.time_meter import TimeMeter
from enum import Enum, auto
import time
import cv2

if __name__ == '__main__':
    api = UpAPI()

    while True:

        frame = api.get_camera_frame()
        cv2.imshow("camera", frame)
        cv2.waitKey(1)