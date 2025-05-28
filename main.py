# 机器人视觉竞技 A

from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.pid import PIDController
from sdk.logic_layer.time_meter import TimeMeter
from robot_body import RobotBody
import time


class Controller:
    def __init__(self):
        # 参数设置
        self.grayscale_threshold = 1600  # 灰度传感器检测阈值

        self.speed_follow_line = 10  # 巡线前进移动速度
        self.move_speed = 8  # 在白色区域前进移动速度
        self.speed_hit_position = 8  # 进入退出击打位置的速度
        self.speed_locate_move = 4  # 定位移动速度
        self.speed_locate_turn = 20  # 定位旋转速度
        self.speed_spin = 50  # 自旋速度
        self.speed_aim = 4  # 瞄准移动速度 TBD
        self.speed_depth_adjust = 3  # 纵向位置调整速度

        self.time_init = 1000  # 初始化时间，单位毫秒
        self.time_left_turn_s = 3.8  # 向左转时间，单位秒
        self.time_right_turn_s = 3.360  # 向右转时间，单位秒
        self.time_backward_turn_s = 7.160  # 向后转时间，单位秒
        self.time_arm_action = 2500  # 手臂做动作时间，单位毫秒
        self.time_enter_home = 1500  # 巡线结束到停车之间的时间，单位毫秒
        self.time_hit_position = 1000  # 进入退出击打状态的移动时间，单位毫秒
        self.time_back_short = 800  # 重定位短距离后退，单位毫秒
        self.time_back_long = 1900  # 重定位长距离后退，单位毫秒

        self.k_p = 15  # 巡线比例参数
        self.k_i = 0.2  # 巡线积分参数
        self.k_d = 1.0  # 巡线微分参数

        self.target_face = "t_0"  # 人脸识别标签
        self.target_vehicle = "tank"  # 车辆识别标签
        self.target_id = 1  # April Tag 识别 ID  （实际是 1 号）
        self.target_number_right = 0  # 手势识别数字，举右手
        self.target_number_both = 5  # 手势识别数字，举右手

        self.target_center_offset = 35  # 目标中心与屏幕中心偏移量，单位像素 TBD
        self.target_width_vehicle = 200  # 车辆目标期望宽度，单位像素 TBD
        self.target_width_face = 150  # 人脸目标期望宽度，单位像素 TBD
        self.target_width_tolerance = 20  # 目标宽度容差，单位像素 TBD

        # YOLO 目标参数
        self.yolo_model = YoloModel.VEHICLE

        # 手臂动作
        self.left_arm_actions = {
            "clamp": arm_action.left_arm_clamp(),
            "up": arm_action.left_arm_raise()
        }
        self.right_arm_actions = {
            "clamp": arm_action.right_arm_clamp(),
            "up": arm_action.right_arm_raise(),
            "pre_hit": arm_action.right_arm_prepare_beat(),
            "hit": arm_action.right_arm_beat()
        }

        # 传感器和执行器
        self.api = UpAPI(yolo_model=self.yolo_model, grayscale_threshold=self.grayscale_threshold)
        
        # 机器人身体控制
        self.robot_body = RobotBody(self.api)

        # 逻辑处理器
        self.locator = CrossLocator()
        self.pid = PIDController(k_p=self.k_p, k_i=self.k_i, k_d=self.k_d)

        # 计时器
        self.initializer = TimeMeter(self.time_init)  # 初始化
        self.timer_arm_action = TimeMeter(self.time_arm_action)  # 手臂做动作
        self.timer_enter_home = TimeMeter(self.time_enter_home)  # 回家
        
        # 相机稳定
        self.count_stable = 0  # 相机稳定计数器
        self.count_continuous_stable = 25  # 相机连续稳定阈值
        
        # 结束巡线的次数
        self.count_out_line = 0
        self.count_max_out_line = 5

    def run(self):
        # 1. 初始化
        self.__initialize()
        
        # 2. April Tag识别流程
        self.robot_body.navigate_to_position_april_tag()
        self.__recognize_and_act_apriltag()
        
        # 3. 手势识别流程
        self.robot_body.navigate_to_position_gesture()
        self.__recognize_and_act_gesture()
        
        # 4. Vehicle识别流程
        self.robot_body.navigate_to_position_vehicle()
        self.__aim_yolo_target_and_adjust()
        self.__recognize_and_act_vehicle()
        
        # 5. 人脸识别流程
        self.robot_body.navigate_to_position_face()
        self.__aim_face_target_and_adjust()
        self.__recognize_and_act_face()
        
        # 6. 回家
        self.robot_body.navigate_to_position_home()
        
        # 7. 完成
        self.__finish()

    def __initialize(self):
        """初始化机器人"""
        print("初始化中...")
        self.__clamp_arms()
        
        # 等待初始化完成
        time.sleep(self.time_init / 1000)
        print("初始化完成")

    def __recognize_and_act_apriltag(self):
        """识别April Tag并做出相应动作"""
        print("演习区域，准备识别 April Tag")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        # 开始识别并执行动作
        self.timer_arm_action.start()
        start_time = time.time()
        
        while time.time() - start_time < self.time_arm_action / 1000:
            find_tag, tag_id, offset_x = self.api.detect_apriltag()
            if find_tag:
                print(f"找到 April Tag 动作：{tag_id}")
                self.__do_arm_action(tag_id)
            time.sleep(0.1)  # 小延时避免CPU占用过高
        
        print("April Tag 动作完成")
        self.__clamp_arms()
        self.api.close_tag_window()

    def __recognize_and_act_gesture(self):
        """识别手势并做出相应动作"""
        print("演习区域，准备识别手势图像")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        # 开始识别并执行动作
        self.timer_arm_action.start()
        start_time = time.time()
        
        while time.time() - start_time < self.time_arm_action / 1000:
            find_target, number = self.api.detect_gesture()
            if find_target:
                print(f"找到手势动作：{number}")
                self.__do_arm_action(number)
            time.sleep(0.1)  # 小延时避免CPU占用过高
        
        print("手势动作完成")
        self.__clamp_arms()
        self.api.close_gesture_window()

    def __recognize_and_act_vehicle(self):
        """识别vehicle目标并击打"""
        print("vehicle 识别区域，预加载图像中")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        # 预加载YOLO图像
        preload_complete = self.api.preload_yolo_pool()
        if preload_complete:
            print("预加载图像完成，准备瞄准")
            
            # 执行击打动作
            print(f"开始击打 vehicle 目标：{self.target_vehicle}")
            self.__hit_actions()
            
            # 清理vehicle资源
            self.__reset_vehicle()

    def __recognize_and_act_face(self):
        """识别人脸并击打"""
        print("人脸识别区域，准备瞄准")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        # 执行击打动作
        print(f"开始击打人脸目标：{self.target_face}")
        self.__hit_actions()
        
        # 关闭人脸窗口
        self.api.close_face_window()

    def __finish(self):
        """完成所有任务"""
        print("任务完成")
        self.api.stop()

    def __aim_yolo_target_and_adjust(self):
        """瞄准YOLO目标"""
        while True:
            find_target, offset_x, width = self.api.detect_yolo(label=self.target_vehicle)
            if find_target:
                # 计算宽度偏移
                offset_width = width - self.target_width_vehicle
                if self.robot_body.adjust_position(offset_x, offset_width):
                    break
            time.sleep(0.1)  # 小延时避免CPU占用过高

    def __aim_face_target_and_adjust(self):
        """瞄准人脸目标"""
        while True:
            find_target, offset_x, width = self.api.detect_face(label=self.target_face)
            if find_target:
                # 计算宽度偏移
                offset_width = width - self.target_width_face
                if self.robot_body.adjust_position(offset_x, offset_width):
                    break
            time.sleep(0.1)  # 小延时避免CPU占用过高

    def __wait_camera_stable(self):
        """等待相机稳定"""
        self.count_stable = 0
        while self.count_stable < self.count_continuous_stable:
            print("等待相机稳定")
            self.api.stop()
            self.count_stable += 1
            self.__clamp_arms()
            time.sleep(0.1)  # 小延时

    def __follow_line(self, offset):
        """沿线行走"""
        turn_rate = self.pid.compute(offset)
        self.api.move_rotation(speed=self.speed_follow_line, turn_rate=turn_rate)

    def __hit_actions(self):
        """执行击打动作序列"""
        # 前进
        self.robot_body.move_distance("forward", 0.2)  # 前进约20厘米

        # 击打动作序列
        self.__pre_hit()
        time.sleep(self.time_arm_action / 1000)
        self.__hit()
        time.sleep(self.time_arm_action / 1000)
        self.__clamp_arms()

        # 后退
        self.robot_body.move_distance("backward", 0.2)  # 后退约20厘米

    def __do_arm_action(self, number):
        """根据识别结果执行手臂动作"""
        if number == self.target_id:
            # 举左手
            left_action = self.left_arm_actions["up"]
            right_action = self.right_arm_actions["clamp"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == self.target_number_right:
            # 举右手
            left_action = self.left_arm_actions["clamp"]
            right_action = self.right_arm_actions["up"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == self.target_number_both:
            # 举双手
            left_action = self.left_arm_actions["up"]
            right_action = self.right_arm_actions["up"]
            self.api.execute_arm_action(left_action, right_action)

    def __clamp_arms(self):
        """夹紧手臂"""
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["clamp"]
        self.api.execute_arm_action(left_action, right_action)

    def __pre_hit(self):
        """准备击打姿势"""
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["pre_hit"]
        self.api.execute_arm_action(left_action, right_action)

    def __hit(self):
        """执行击打动作"""
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["hit"]
        self.api.execute_arm_action(left_action, right_action)

    def __reset_vehicle(self):
        """重置YOLO资源"""
        self.api.close_yolo_window()
        self.api.reset_yolo_pool()


if __name__ == '__main__':
    controller = Controller()
    controller.run()
