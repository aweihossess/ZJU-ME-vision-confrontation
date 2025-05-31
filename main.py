# 机器人视觉竞技 A

from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.time_meter import TimeMeter
from robot_body import RobotBody
import time

TARGET_FACE = "t_0"  # 人脸识别标签
TARGET_VEHICLE = "tank"  # 车辆识别标签
TARGET_ID = 1  # April Tag 识别 ID  （实际是 1 号）
TARGET_NUMBER_RIGHT = 0  # 手势识别数字，举右手
TARGET_NUMBER_BOTH = 5  # 手势识别数字，举右手

class Controller:
    def __init__(self):
        # 参数设置
        self.grayscale_threshold = 1600  # 灰度传感器检测阈值

        self.time_init = 1000  # 初始化时间，单位毫秒
        self.arm_action_duration = 2500  # 手臂做动作时间，单位毫秒 TBD
        self.arm_action_delay = 0.1  # 手臂做动作时间，单位秒
        self.time_hit_position = 1000  # 进入退出击打状态的移动时间，单位毫秒 TBD

        self.target_width_vehicle = 160  # 车辆目标期望宽度，单位像素
        self.target_width_face = 160  # 人脸目标期望宽度，单位像素 TBD

        # YOLO 目标参数
        self.yolo_model = YoloModel.VEHICLE

        # 手臂动作的对应关系映射
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

        # 计时器
        self.timer_arm_action = TimeMeter(self.arm_action_duration)  # 手臂做动作
        
        # 相机稳定
        self.count_stable = 0  # 相机稳定计数器
        self.count_continuous_stable = 5  # 相机连续稳定阈值

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
        offset_x, ratio_w = self.__recognize_yolo_target()
        self.__adjust_and_act_vehicle(offset_x, ratio_w)
        self.robot_body.adjust_position(-offset_x, 2-ratio_w, target_type="vehicle")
        
        # # 5. 人脸识别流程
        self.robot_body.navigate_to_position_face()
        offset_x, ratio_w = self.__recognize_face_target()
        self.__adjust_and_act_face(offset_x, ratio_w)
        self.robot_body.adjust_position(-offset_x, 2-ratio_w, target_type="face")
        
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
        
        while time.time() - start_time < self.arm_action_duration / 1000:
            find_tag, tag_id, offset_x = self.api.detect_apriltag()
            if find_tag:
                print(f"找到 April Tag 动作：{tag_id}")
                self.__do_arm_action(tag_id)
                time.sleep(1)
                break
            time.sleep(self.arm_action_delay)  # 小延时避免CPU占用过高
        
        print("April Tag 动作完成")
        self.__clamp_arms()
        self.api.close_tag_window()

    def __recognize_and_act_gesture(self):
        """识别手势并做出相应动作"""
        print("演习区域，准备识别手势图像")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        detect_gesture_true = False
        
        # 开始识别并执行动作
        self.timer_arm_action.start()
        start_time = time.time()
        
        while time.time() - start_time < self.arm_action_duration / 1000:
            find_target, number = self.api.detect_gesture()
            if find_target:
                print(f"找到手势动作：{number}")
                self.__do_arm_action(number)
                time.sleep(1)
                detect_gesture_true = True
                break
            time.sleep(self.arm_action_delay)  # 小延时避免CPU占用过高
        
        if not detect_gesture_true:
            print("未识别到手势动作")
        
        print("手势动作完成")
        self.__clamp_arms()
        self.api.close_gesture_window()

    def __finish(self):
        """完成所有任务"""
        print("任务完成")
        self.api.stop()

    def __recognize_yolo_target(self):
        """识别YOLO目标并返回偏移量"""
        print("vehicle 识别区域，预加载图像中")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        self.timer_arm_action.start()
        start_time = time.time()
        
        while time.time() - start_time < self.arm_action_duration / 1000:
            # 预加载YOLO图像
            preload_complete = self.api.preload_yolo_pool()
            if not preload_complete:
                print("预加载图像失败")
                time.sleep(self.arm_action_delay)  # 小延时避免CPU占用过高
                continue
            
            print("预加载图像完成，准备识别目标")

            # 开始识别目标
            find_target, offset_x, width = self.api.detect_yolo(label=TARGET_VEHICLE)
            if not find_target:
                print("未找到目标，继续寻找目标")
                time.sleep(self.arm_action_delay)  # 小延时避免CPU占用过高
                continue

            print(f"找到 vehicle 目标：{TARGET_VEHICLE}")
            # 计算宽度偏移
            print(f"width={width}")
            ratio_w = width / self.target_width_vehicle
            return offset_x, ratio_w
        
        print("未在时间内识别到目标")
        return 0, 0  # 如果没有识别到目标，返回零偏移

    def __adjust_and_act_vehicle(self, offset_x, ratio_w):
        """调整位置并击打vehicle目标"""
        # 调整位置
        if offset_x != 0 or ratio_w != 0:
            self.robot_body.adjust_position(offset_x, ratio_w, target_type="vehicle")
        
        # 执行击打动作
        print(f"开始击打 vehicle 目标：{TARGET_VEHICLE}")
        self.__hit_actions()
        
        # 清理vehicle资源
        self.__reset_vehicle()

    def __recognize_face_target(self):
        """识别人脸目标并返回偏移量"""
        print("人脸识别区域，准备识别目标")
        
        # 等待相机稳定
        self.__wait_camera_stable()
        
        # 开始识别目标
        self.timer_arm_action.start()
        start_time = time.time()
        
        while time.time() - start_time < self.arm_action_duration / 1000:
            find_target, offset_x, width = self.api.detect_face(label=TARGET_FACE)
            print(f"find_target={find_target}, offset_x={offset_x}, width={width}")
            if find_target:
                print(f"offset_x={offset_x}, width={width}")
                print(f"找到人脸目标：{TARGET_FACE}")
                # 计算宽度偏移
                ratio_w = width / self.target_width_face
                print(f"ratio_w={ratio_w}")
                return offset_x, ratio_w
            time.sleep(self.arm_action_delay)  # 小延时避免CPU占用过高
        
        print("未在时间内识别到人脸目标")
        return 0, 0  # 如果没有识别到目标，返回零偏移

    def __adjust_and_act_face(self, offset_x, ratio_w):
        """调整位置并击打人脸目标"""
        # 调整位置
        if offset_x != 0 or ratio_w != 0:
            print(f"调整位置: offset_x={offset_x}, ratio_w={ratio_w}")
            self.robot_body.adjust_position(offset_x, ratio_w, target_type="face")
        
        # 执行击打动作
        print(f"开始击打人脸目标：{TARGET_FACE}")
        self.__hit_actions()
        
        # 关闭人脸窗口
        self.api.close_face_window()

    def __wait_camera_stable(self):
        """等待相机稳定"""
        self.__clamp_arms()
        self.api.stop()
        self.count_stable = 0
        while self.count_stable < self.count_continuous_stable:
            print("等待相机稳定")
            self.count_stable += 1
            time.sleep(0.1)  # 小延时

    def __hit_actions(self):
        """执行击打动作序列"""
        # 击打动作序列
        self.__pre_hit()
        time.sleep(self.time_hit_position / 1000)
        self.__hit()
        time.sleep(self.time_hit_position / 1000)
        self.__clamp_arms()
        time.sleep(self.time_hit_position / 1000)

    def __do_arm_action(self, number):
        """根据识别结果执行手臂动作"""
        if number == TARGET_ID:
            # 举左手
            left_action = self.left_arm_actions["up"]
            right_action = self.right_arm_actions["clamp"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == TARGET_NUMBER_RIGHT:
            # 举右手  
            left_action = self.left_arm_actions["clamp"]
            right_action = self.right_arm_actions["up"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == TARGET_NUMBER_BOTH:
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
