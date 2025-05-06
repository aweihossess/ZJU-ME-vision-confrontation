from sdk.data_layer.robot import Robot
from sdk.hardware_layer.manager.serial_manager import SerialManager
import time


if __name__ == '__main__':
    serial_manager = SerialManager()
    serial_port = serial_manager.get_serial_usb()
    robot = Robot(serial_port)

    print("关闭双闭环")
    robot.set_control_model(False)
    time.sleep(1)
    
    print("开始校准IMU")
    robot.calibrate_imu()
    time.sleep(12)
    print("IMU校准完成")

    print("程序结束...")
