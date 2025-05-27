from ..data_layer.arm.arm_data import UpperBody
from ..hardware_layer.manager.serial_manager import SerialManager
from ..data_layer.robot import Robot
from .notice.base import NoticeBase
import time


class Action(NoticeBase):
    def __init__(self):
        super().__init__()

        # 实例化串口管理
        self.serial_manager = SerialManager()

        # 实例化机器人通讯
        self.robot = Robot(self.serial_manager.get_serial_uart())

        # 开启双闭环模式
        self.robot.set_control_model(True)

    def clean_up(self):
        print("重置底盘模式...")
        if self.serial_manager.get_serial_uart().is_open:
            self.robot.move(angle=0, speed=0, turn_rate=0, execute_time=10)
            time.sleep(0.03)
            self.robot.set_control_model(False)
            time.sleep(0.03)
            self.serial_manager.get_serial_uart().close()
        if self.serial_manager.get_serial_usb().is_open:
            self.serial_manager.get_serial_usb().close()

    # 底盘移动

    def move_translation(self, angle=0, speed=10, run_time=10):
        """
        水平移动

        :param angle: 平移角度  0~360  (正前方为0°，逆时针为正方向)
        :param speed: 移动速度  0~100
        :param run_time: 执行时间
        """
        self.robot.move(angle, speed, 0, run_time)

    def move_rotation(self, speed=10, turn_rate=0, run_time=10):
        """
        转向

        :param speed: 移动速度  0~100
        :param turn_rate: 旋转速率  Mode为0、1时取值范围：-100~100，Mode为2时取值范围：-1000~1000
        :param run_time: 执行时间
        """
        self.robot.move(0, speed, turn_rate, run_time)

    def seeking(self, turn_rate=100, run_time=10):
        """
        原地自旋

        :param turn_rate: 旋转速率  Mode为0、1时取值范围：-100~100，Mode为2时取值范围：-1000~1000
        :param run_time: 执行时间
        """
        self.robot.move(0, 0, turn_rate, run_time)

    # 手臂控制

    def set_servo_positions(self, left_arm_positions, right_arm_position, run_time=10):
        """
        设置舵机位置

        :param left_arm_positions: 左臂舵机位置
        :param right_arm_position: 右臂舵机位置
        :param run_time: 执行时间
        """
        body_data = UpperBody(left_arm_positions, right_arm_position, run_time)
        body_data_list = body_data.to_list()

        self.robot.set_servo_positions(body_data_list)
