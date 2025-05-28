import time
from sdk.api import UpAPI
import math


class RobotBody:
    def __init__(self, api):
        """初始化机器人控制接口"""
        self.api = api
        # 机器人的运动参数
        self.default_speed = 50  # 默认速度
        self.default_turn_rate = 200  # 默认旋转速率
        # 机器人的物理参数（需要根据实际情况调整）
        self.wheel_radius = 0.05  # 轮子半径（米）
        self.robot_speed_factor = 0.005  # 速度转换因子（speed=50时每秒移动的米数）

        self.max_move_speed = 100 # 最大移动速度
        self.max_rotation_speed = 100 # 最大旋转速度
        # TBD: 待测试给出准确值
        self.rotation_90_time = 1800 # 90度旋转时间
        self.extra_sleep_delay = 0.5  # 移动/旋转后额外休眠时间（秒），确保移动/旋转完全完成
        
        # 目标瞄准参数
        self.target_center_offset = 35  # 目标中心与屏幕中心偏移量，单位像素
        self.target_width_tolerance = 20  # 目标宽度容差，单位像素
    
    def move_forward(self, move_time):
        """
        使机器人向前移动
        
        :param move_time: 移动时间（毫秒）
        """
        self.api.move_forward(speed=self.max_move_speed, run_time=move_time)
        time.sleep(move_time / 1000 + self.extra_sleep_delay)
        self.api.stop()
    
    def move_backward(self, move_time):
        """
        使机器人向后移动
        
        :param move_time: 移动时间（毫秒）
        """
        self.api.move_backward(speed=self.max_move_speed, run_time=move_time)
        time.sleep(move_time / 1000 + self.extra_sleep_delay)
        self.api.stop()

    def move_left_forward(self):
        """
        使机器人往左前方angle度方向直线移动
        
        :param angle: 移动角度（度）
        :return: 无
        """
        angle = 45 # TBD
        move_time = 2300 # TBD
        self.api.move_translation(angle=angle, speed=self.max_move_speed, run_time=move_time)
        time.sleep(move_time / 1000 + self.extra_sleep_delay)

    def move_right_forward(self):
        """
        使机器人往右前方angle度方向直线移动
        
        :param angle: 移动角度（度）
        :return: 无
        """
        angle = -45 # TBD
        move_time = 2300 # TBD
        self.api.move_translation(angle=angle, speed=self.max_move_speed, run_time=move_time)
        time.sleep(move_time / 1000 + self.extra_sleep_delay)
        self.api.stop()

    def rotate_90_degrees(self, direction="left"):
        """
        使机器人转90度
        
        :param direction: 旋转方向，"left"为左转，"right"为右转
        """
        if direction == "left":
            self.api.spin_left(self.max_rotation_speed, self.rotation_90_time)
        elif direction == "right":
            self.api.spin_right(self.max_rotation_speed, self.rotation_90_time)
        else:
            print("无效的旋转方向，请指定'left'或'right'")
            return
        time.sleep(self.rotation_90_time / 1000 + self.extra_sleep_delay)
        self.api.stop()

    def rotate_left_180_degrees(self):
        """
        使机器人左转180度
        """
        self.api.spin_left(self.max_rotation_speed, self.rotation_90_time * 2)
        time.sleep(self.rotation_90_time * 2 / 1000 + self.extra_sleep_delay)

    def navigate_to_position_april_tag(self):
        """导航到April Tag识别位置"""
        print("导航到April Tag识别位置")
        # 往左前方直接移动到april tag

        self.move_left_forward()
        self.rotate_90_degrees("left")
        print("到达 April Tag 识别十字")

    def navigate_to_position_gesture(self):
        """导航到手势识别位置"""
        print("导航到手势识别位置")
        # 往后转，再移动到手掌位置
        self.rotate_left_180_degrees()
        move_time = 1000 # TBD
        self.move_forward(move_time)
        print("到达手势识别十字")

    def navigate_to_position_vehicle(self):
        """导航到vehicle识别位置"""
        print("导航到vehicle识别位置")
        # 往后转，再向右前方移动到车辆位置
        self.rotate_left_180_degrees()
        self.move_right_forward()
        print("到达 vehicle 识别")

    def navigate_to_position_face(self):
        """导航到人脸识别位置"""
        print("导航到人脸识别位置")
        # 往右转，再向前方移动到人脸位置
        self.rotate_90_degrees("right")
        move_time = 1000 # TBD
        self.move_forward(move_time)
        print("到达人脸识别十字")

    def navigate_to_position_home(self):
        """导航到回家位置"""
        print("导航到回家位置")
        # 往后一直倒退，回到初始位置
        move_time = 1000 # TBD
        self.move_backward(move_time)
        print("准备回家")


    def move_distance(self, direction, distance):
        """
        向指定方向移动指定距离
        
        :param direction: 移动方向，可以是"forward", "backward", "left", "right"或0-359的角度值
        :param distance: 移动距离（米）
        """
        # 计算移动所需的时间（毫秒）
        speed = self.max_move_speed
        move_time = int((distance / (speed * self.robot_speed_factor)) * 1000)
        
        print(f"向{direction}方向移动{distance}米")
        
        # 根据方向选择移动方式
        if direction == "forward":
            self.api.move_forward(speed=speed, run_time=move_time)
        elif direction == "backward":
            self.api.move_backward(speed=speed, run_time=move_time)
        elif direction == "left":
            self.api.move_left(speed=speed, run_time=move_time)
        elif direction == "right":
            self.api.move_right(speed=speed, run_time=move_time)
        elif isinstance(direction, (int, float)) and 0 <= direction < 360:
            # 如果direction是角度值（0-359），使用move_translation
            self.api.move_translation(angle=direction, speed=speed, run_time=move_time)
        else:
            print("无效的移动方向，请指定'forward', 'backward', 'left', 'right'或0-359的角度值")
            return
        
        # 等待移动完成
        time.sleep(move_time / 1000)
        print(f"移动完成，已行进{distance}米")
        
    def adjust_position(self, offset_x, offset_width):
        """
        瞄准目标，返回是否瞄准成功
        
        :param offset_x: 目标横向偏移（像素）
        :param offset_width: 目标宽度偏移（当前宽度 - 期望宽度，像素）
        :return: 是否瞄准成功
        """
        if offset_x is None:
            print("没有发现目标")
            self.api.stop()
            return False
            
        # 横向位置调整
        horizontal_aligned = False
        if abs(offset_x) >= self.target_center_offset:
            # 计算横向移动距离（米），与偏移量成正比
            # 限制最小和最大距离
            distance = min(max(abs(offset_x) / 500, 0.02), 0.1)  # 将像素偏移转换为米
            
            if offset_x > 0:
                # 目标在右侧，需要向右移动
                self.move_distance("right", distance)
                print(f"向右调整，偏移量: {offset_x}，距离: {distance:.3f}米")
            else:
                # 目标在左侧，需要向左移动
                self.move_distance("left", distance)
                print(f"向左调整，偏移量: {offset_x}，距离: {distance:.3f}米")
        else:
            horizontal_aligned = True
            
        # 纵向位置调整
        depth_aligned = False
        if offset_width is not None:
            if abs(offset_width) > self.target_width_tolerance:
                # 计算纵向移动距离（米），与宽度差值成正比
                # 限制最小和最大距离
                distance = min(max(abs(offset_width) / 400, 0.02), 0.1)  # 将像素差值转换为米
                
                if offset_width < 0:
                    # 目标太小，需要前进
                    self.move_distance("forward", distance)
                    print(f"目标太远，前进调整 (宽度偏移: {offset_width}，距离: {distance:.3f}米)")
                else:
                    # 目标太大，需要后退
                    self.move_distance("backward", distance)
                    print(f"目标太近，后退调整 (宽度偏移: {offset_width}，距离: {distance:.3f}米)")
            else:
                depth_aligned = True
                print(f"目标距离合适 (宽度偏移: {offset_width})")
        else:
            # 如果没有宽度信息，则只考虑横向对齐
            depth_aligned = True
            
        # 如果横向和纵向都对齐，则停止移动并返回成功
        if horizontal_aligned and depth_aligned:
            self.api.stop()
            return True
            
        return False


# 测试代码
if __name__ == "__main__":
    try:
        api = UpAPI()
        robot = RobotBody(api)
        
        # 测试旋转90度
        print("navigate_to_position_april_tag")
        robot.navigate_to_position_april_tag()
        print("navigate_to_position_gesture")
        robot.navigate_to_position_gesture()
        print("navigate_to_position_vehicle")
        robot.navigate_to_position_vehicle()
        print("navigate_to_position_face")
        robot.navigate_to_position_face()
        print("navigate_to_position_home")
        robot.navigate_to_position_home()
        
        # # 测试移动指定距离
        # print("\n测试移动功能...")
        # robot.move_distance("forward", 0.5)  # 向前移动0.5米
        # time.sleep(1)
        # robot.move_distance("backward", 0.5)  # 向后移动0.5米
        # time.sleep(1)
        # robot.move_distance("left", 0.3)  # 向左移动0.3米
        # time.sleep(1)
        # robot.move_distance("right", 0.3)  # 向右移动0.3米
        # time.sleep(1)
        # robot.move_distance(45, 0.4)  # 向45度方向移动0.4米
        
        print("测试完成!")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保机器人停止
        if 'robot' in locals():
            robot.api.stop()
            print("机器人已停止") 
