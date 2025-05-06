from ..hardware_layer.communicator.robot_communicator import RobotCommunicator
from ..utils import convert_util as tools
from .communication.robot_data import RobotChassisCommand, RobotServoCommand


class Robot:
    __commands = {
        'chassis_control_model': 0x01,
        'chassis_move': 0x02,
        'chassis_imu_calibration': 0x04,
        'servo_positions': 0x5F,
        'servo_positions_in_file': 0x55,
    }

    def __init__(self, serial_port):
        self.__communicator = RobotCommunicator(serial_port)
        self.double_loop = False  # 是否开启双闭环控制  0——单速度环模式，2——速度角速度双闭环模式

    def __limit_turn_rate(self, turn_rate):
        limit = 1000 if self.double_loop else 100
        return max(-limit, min(turn_rate, limit))

    # 底盘控制

    def set_control_model(self, double_loop):
        """
        设置底盘运动控制模式
        :param double_loop: 是否开启双闭环模式
        """
        mode = 2 if double_loop else 0

        command = RobotChassisCommand(command=self.__commands['chassis_control_model'], parameters=[mode])
        byte_array = command.to_bytes()

        self.__communicator.send_command(byte_array)
        self.double_loop = double_loop

    def move(self, angle, speed, turn_rate, execute_time):
        """
        移动
        :param angle: 平移角度  0~360  (正前方为0°，逆时针为正方向)
        :param speed: 移动速度  0~100
        :param turn_rate: 旋转速率  Mode为0、1时取值范围：-100~100，Mode为2时取值范围：-1000~1000
        :param execute_time: 执行时间
        """
        limit_turn_rate = self.__limit_turn_rate(turn_rate)  # 防止发送超范围

        bytearray_angle = tools.decimal_convert_to_little_endian_list(angle)
        bytearray_speed = tools.decimal_convert_to_little_endian_list(speed)
        bytearray_turn_rate = tools.decimal_convert_to_little_endian_list(limit_turn_rate)
        bytearray_time = tools.decimal_convert_to_little_endian_list(execute_time)
        parameters = bytearray_angle + bytearray_speed + bytearray_turn_rate + bytearray_time

        command = RobotChassisCommand(command=self.__commands['chassis_move'], parameters=parameters)
        byte_array = command.to_bytes()

        self.__communicator.send_command(byte_array)

    def calibrate_imu(self):
        """
        校准 IMU
        """
        command = RobotChassisCommand(command=self.__commands['chassis_imu_calibration'])
        byte_array = command.to_bytes()

        self.__communicator.send_command(byte_array)

    # 舵机控制

    def set_servo_positions(self, servo_position_list):
        command = RobotServoCommand(command=self.__commands['servo_positions'], parameters=servo_position_list)
        byte_array = command.to_bytes()

        self.__communicator.send_command(byte_array)
