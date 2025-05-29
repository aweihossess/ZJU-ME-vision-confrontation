import time
from sdk.api import UpAPI
import math

k_back_error_correction = 0.8

class RobotBody:
    def __init__(self, api):
        """初始化机器人控制接口"""
        self.api = api
        # 机器人的运动参数
        self.default_speed = 100 # 默认就是最大移动速度
        self.default_turn_rate = 400 # 默认即最大旋转速率
        self.speed_factor = 0.5  # 速度转换因子（speed=100时每秒移动的米数）
        self.turn_rate_factor = 0.5  # 旋转速率转换因子（turn_rate=100时每秒旋转的度数）

        # TBD: 待测试给出准确值
        self.rotation_90_time = 450 # 90度旋转时间
        self.rotation_180_time = 900 # 180度旋转时间
        self.extra_sleep_delay = 0.1  # 移动/旋转后额外休眠时间（秒），确保移动/旋转完全完成
        self.k_vehicle_horizontal = 1.56  # 140mm对应着90个像素，所以1个像素对应1.56mm
        self.k_vehicle_vertical = 0.5 # 由distance = self.k_vertical * (1 - ratio_w)反推计算得到
        self.k_face_horizontal = 1.2
        self.k_face_vertical = 0.4 # 由distance = self.k_vertical * (1 - ratio_w)反推计算得到
        
        # 目标瞄准参数
        self.target_x_offset_tolerance = 35  # 目标中心与屏幕中心偏移量，单位像素 TBD
        self.target_width_tolerance = 0.05  # 目标宽度容差
    
    def move_forward(self, distance):
        """
        使机器人向前移动
        
        :param distance: 移动距离（米）
        """
        # 计算移动目标距离所需的时间（毫秒）
        run_time = int((distance / self.speed_factor) * 1000)
        
        self.api.move_forward(speed=self.default_speed, run_time=run_time)
        time.sleep(run_time / 1000 + self.extra_sleep_delay)
        self.api.stop()
    
    def move_backward(self, distance):
        """
        使机器人向后移动
        
        :param distance: 移动距离（米）
        """
        distance = distance * k_back_error_correction
        # 计算移动目标距离所需的时间（毫秒）
        run_time = int((distance / self.speed_factor) * 1000)
        
        self.api.move_backward(speed=self.default_speed, run_time=run_time)
        time.sleep(run_time / 1000 + self.extra_sleep_delay)
        self.api.stop()

    def rotate_90_degrees(self, direction="left"):
        """
        使机器人转90度
        
        :param direction: 旋转方向，"left"为左转，"right"为右转
        """
        if direction == "left":
            self.api.spin_left(turn_rate=self.default_turn_rate, run_time=self.rotation_90_time)
        elif direction == "right":
            self.api.spin_right(turn_rate=self.default_turn_rate, run_time=self.rotation_90_time)
        else:
            print("无效的旋转方向，请指定'left'或'right'")
            return
        time.sleep(self.rotation_90_time / 1000 + self.extra_sleep_delay)
        self.api.stop()

    def rotate_left_180_degrees(self):
        """
        使机器人左转180度
        """
        self.api.spin_left(turn_rate=self.default_turn_rate, run_time=self.rotation_180_time)
        time.sleep(self.rotation_180_time / 1000 + self.extra_sleep_delay)

    def navigate_to_position_april_tag(self):
        """导航到April Tag识别位置"""
        print("导航到April Tag识别位置")
        # 往左前方直接移动到april tag
        distance = 0.6
        self.move_forward(distance)
        self.rotate_90_degrees("left")
        self.move_forward(distance)
        print("到达 April Tag 识别十字")

    def navigate_to_position_gesture(self):
        """导航到手势识别位置"""
        print("导航到手势识别位置")
        # 往后转，再移动到手掌位置
        self.rotate_left_180_degrees()
        distance = 1.2  # 移动1.2米，即0.6*2
        self.move_forward(distance)
        print("到达手势识别十字")

    def navigate_to_position_vehicle(self):
        """导航到vehicle识别位置"""
        print("导航到vehicle识别位置")
        # 往后转，再向右前方移动到车辆位置
        self.rotate_left_180_degrees()
        distance = 0.55
        self.move_forward(distance)
        self.move_distance("right", distance)
        # distance = 0.75  # 移动0.84米，还是0.6*sqrt(2)
        # self.move_right_forward(distance)
        print("到达 vehicle 识别")

    def navigate_to_position_face(self):
        """导航到人脸识别位置"""
        print("导航到人脸识别位置")
        # 往右转，再向前方移动到人脸位置
        self.rotate_90_degrees("right")
        distance = 0.6
        self.move_forward(distance)
        print("到达人脸识别十字")

    def navigate_to_position_home(self):
        """导航到回家位置"""
        print("导航到回家位置")
        # 往后一直倒退，回到初始位置
        distance = 1.9  # 移动1.8米，即0.6*3
        self.move_backward(distance)
        print("准备回家")


    def move_distance(self, direction, distance):
        """
        向指定方向移动指定距离
        
        :param direction: 移动方向，可以是"forward", "backward", "left", "right"或0-359的角度值
        :param distance: 移动距离（米）
        """
        # 计算移动所需的时间（毫秒）
        speed = self.default_speed
        move_time = int((distance / self.speed_factor) * 1000)
        
        print(f"向{direction}方向移动{distance}米")
        
        # 根据方向选择移动方式
        if direction == "forward":
            self.api.move_forward(speed=speed, run_time=move_time)
        elif direction == "backward":
            move_time = move_time * k_back_error_correction
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
        time.sleep(move_time / 1000 + self.extra_sleep_delay)
        self.api.stop()
        print(f"移动完成，已行进{distance}米")
        
    def adjust_position(self, offset_x, ratio_w, target_type):
        """
        瞄准目标，返回是否瞄准成功
        
        :param offset_x: 目标横向偏移（像素），正值表示目标在右侧，负值表示目标在左侧
        :param ratio_w: 目标宽度比例（当前宽度 / 期望宽度）
        :return: 是否调整成功
        """
        k_horizontal = self.k_vehicle_horizontal if target_type == "vehicle" else self.k_face_horizontal
        k_vertical = self.k_vehicle_vertical if target_type == "vehicle" else self.k_face_vertical
        print("*"*50)
        print(f"调整位置: offset_x={offset_x}, ratio_w={ratio_w}")
        adjusted = False
        
        # 横向位置调整 - 根据公式：distance_x = kx · (x - x0) = kx · offset_x
        if abs(offset_x) >= self.target_x_offset_tolerance:
            # 计算横向移动距离（米），与横向偏移像素成正比
            distance = k_horizontal * offset_x / 1000  # 转换为米
            distance = min(max(abs(distance), 0.02), 0.25)  # 限制最小和最大距离
            print(f"horizontal distance={distance}")
            
            # 当目标在右侧(offset_x > 0)，需要向左移动(distance < 0)
            if offset_x > 0:
                # 目标在右侧，需要向右移动
                self.move_distance("right", distance)
                print(f"向右调整，偏移量: {offset_x}，距离: {distance:.3f}米")
            else:
                # 目标在左侧，需要向左移动
                self.move_distance("left", abs(distance))
                print(f"向左调整，偏移量: {offset_x}，距离: {abs(distance):.3f}米")
            adjusted = True
        else:
            print(f"横向位置已对准 (偏移: {offset_x})")
            
        # 纵向位置调整 - 根据公式：distance_y = ky · (1 - w/w0) = (1 - ratio_w)
        if abs(ratio_w - 1) > self.target_width_tolerance:
            # 计算纵向移动距离（米），与纵向宽度比例成反比
            distance = k_vertical * (1 - ratio_w)
            print(f"vertical distance={distance}")
            distance = min(max(abs(distance), 0.02), 0.2)  # 限制最小和最大距离
            
            # 当 ratio_w < 1 (目标较小)，distance > 0，需要前进
            if ratio_w < 1:
                self.move_distance("forward", abs(distance))
                print(f"目标太远，前进调整 (宽度比例: {ratio_w}，距离: {abs(distance):.3f}米)")
            else:
                # 当 ratio_w > 1 (目标较大)，distance < 0，需要后退
                self.move_distance("backward", abs(distance))
                print(f"目标太近，后退调整 (宽度比例: {ratio_w}，距离: {abs(distance):.3f}米)")
            adjusted = True
        else:
            print(f"纵向位置已对准 (宽度比例: {ratio_w})")
        print("*"*50)
            
        return adjusted


# 测试代码
if __name__ == "__main__":
    try:
        api = UpAPI()
        robot = RobotBody(api)
        
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
