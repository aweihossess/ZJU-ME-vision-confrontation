# 机器人视觉竞技 A
# 这是一个基于状态机的机器人控制系统，用于视觉竞技比赛

from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.pid import PIDController
from sdk.logic_layer.time_meter import TimeMeter
from enum import Enum, auto
import time


# 主状态枚举 - 定义机器人主要行为状态
class MainState(Enum):
    IDLE = auto()        # 空闲/初始化状态
    TRANSITION = auto()  # 转场状态(在不同区域间移动)
    LOCATION = auto()    # 定位状态(寻找十字路口)
    RELOCATION = auto()  # 重定位状态(调整位置)
    RECOGNITION = auto() # 识别状态(执行视觉任务)
    LINE = auto()        # 巡线状态(跟随黑线)
    HOME = auto()        # 回家状态(返回起点)
    FINISH = auto()      # 完成状态


# 目标状态枚举 - 定义机器人当前要执行的任务类型
class TargetState(Enum):
    APRIL_TAG = auto()   # AprilTag识别任务
    GESTURE = auto()     # 手势识别任务
    YOLO = auto()        # YOLO目标检测任务
    FACE = auto()        # 人脸识别任务
    BACK_HOME = auto()   # 返回起点任务


# 转场子状态枚举 - 定义转场过程中的具体行为
class TransitionState(Enum):
    EXIT_CROSS = auto()   # 驶出十字路口
    MOVE_FORWARD = auto() # 在白色区域前进
    SPAN = auto()         # 旋转调整方向


# 重定位子状态枚举 - 定义重定位的具体行为
class RelocationState(Enum):
    LONG = auto()    # 长距离后退(用于YOLO和人脸检测区域)
    SHORT = auto()   # 短距离后退(用于其他区域)
    COMPLETE = auto()# 重定位完成


# 识别子状态枚举 - 定义识别过程中的具体行为
class RecognitionState(Enum):
    TURN_LEFT = auto() # 左转(仅YOLO检测前使用)
    PREPARE = auto()   # 准备阶段
    AIM = auto()       # 瞄准目标
    EXECUTE = auto()   # 执行动作


# 旋转子状态枚举 - 定义旋转的具体方向
class SpanState(Enum):
    FORWARD = auto()   # 向前
    BACKWARD = auto()  # 向后
    LEFT = auto()      # 向左
    RIGHT = auto()     # 向右


class Controller:
    def __init__(self):
        # 参数设置
        self.grayscale_threshold = 3240  # 灰度传感器检测阈值

        self.speed_follow_line = 10  # 巡线前进移动速度
        self.speed_move_in_white = 8  # 在白色区域前进移动速度
        self.speed_hit_position = 14  # 进入退出击打位置的速度
        self.speed_locate_move = 4  # 定位移动速度
        self.speed_locate_turn = 20  # 定位旋转速度
        self.speed_spin = 100  # 自旋速度
        self.speed_aim = 4  # 瞄准移动速度

        self.time_init = 1000  # 初始化时间，单位毫秒
        self.time_left_turn = 1900  # 向左转时间，单位毫秒
        self.time_right_turn = 1680  # 向右转时间，单位毫秒
        self.time_backward_turn = 3580  # 向后转时间，单位毫秒
        self.time_arm_action = 2500  # 手臂做动作时间，单位毫秒
        self.time_enter_home = 1500  # 巡线结束到停车之间的时间，单位毫秒
        self.time_hit_position = 1000  # 进入退出击打状态的移动时间，单位毫秒
        self.time_back_short = 800  # 重定位短距离后退，单位毫秒
        self.time_back_long = 1900  # 重定位长距离后退，单位毫秒

        self.k_p = 16                   # 巡线比例参数
        self.k_i = 0.01                 # 巡线积分参数
        self.k_d = 170                  # 巡线微分参数

        self.target_face = "t_0"  # 人脸识别标签
        self.target_yolo = "tank"  # 车辆识别标签
        self.target_id = 3  # April Tag 识别 ID  （实际是 1 号）
        self.target_number_right = 0  # 手势识别数字，举右手
        self.target_number_both = 5  # 手势识别数字，举右手

        self.target_center_offset = 35  # 目标中心与屏幕中心偏移量，单位像素

        # YOLO 目标参数
        self.yolo_model = YoloModel.VEHICLE

        # 手臂动作
        self.left_arm_actions = {
            "clamp": arm_action.left_arm_clamp(),  # 左手夹紧
            "up": arm_action.left_arm_raise()      # 左手举起
        }
        self.right_arm_actions = {
            "clamp": arm_action.right_arm_clamp(),
            "up": arm_action.right_arm_raise(),
            "pre_hit": arm_action.right_arm_prepare_beat(),
            "hit": arm_action.right_arm_beat()
        }

        # 状态机初始化
        self.state_main = MainState.IDLE          # 主状态
        self.state_target = TargetState.APRIL_TAG # 目标状态
        self.state_transition = TransitionState.EXIT_CROSS
        self.state_relocation = RelocationState.COMPLETE
        self.state_recognition = RecognitionState.PREPARE
        self.state_span = SpanState.FORWARD

        # 传感器和执行器
        self.api = UpAPI(yolo_model=self.yolo_model, grayscale_threshold=self.grayscale_threshold)

        # 逻辑处理器
        self.locator = CrossLocator()
        self.pid = PIDController(k_p=self.k_p, k_i=self.k_i, k_d=self.k_d)

        # 计时器
        self.initializer = TimeMeter(self.time_init)  # 初始化
        self.spanner_left = TimeMeter(self.time_left_turn)  # 向左转
        self.spanner_right = TimeMeter(self.time_right_turn)  # 向右转
        self.spanner_backward = TimeMeter(self.time_backward_turn)  # 向后转
        self.timer_arm_action = TimeMeter(self.time_arm_action)  # 手臂做动作
        self.timer_enter_home = TimeMeter(self.time_enter_home)  # 回家
        self.timer_back_short = TimeMeter(self.time_back_short)  # 短距离重定位
        self.timer_back_long = TimeMeter(self.time_back_long)  # 长距离重定位

        # 相机稳定
        self.count_stable = 0  # 相机稳定计数器
        self.count_continuous_stable = 25  # 相机连续稳定阈值

        # 已经定位十字的次数
        self.count_cross_pass = 0  # 十字定位次数计数器 十字路口通过计数器

        # 结束巡线的次数
        self.count_out_line = 0
        self.count_max_out_line = 5

        # 定位旋转超出预计的次数
        self.count_locate_spin_left = 0
        self.count_locate_spin_right = 0
        self.count_max_locate_spin = 25

    def run(self):
        """主循环，根据当前状态执行相应操作"""
        while True:
            # 获取传感器数据
            grayscale_data = self.api.get_grayscale_data()
            line_center_offset = self.api.follow_line()
            
            # 状态机处理
            if self.state_main == MainState.IDLE:
                # 初始化状态处理
                if self.initializer.complete():
                    print("初始化完成")
                    self.state_main = MainState.TRANSITION
                    self.state_transition = TransitionState.EXIT_CROSS
                else:
                    print("初始化中...")
                    self.__clamp_arms()

            elif self.state_main == MainState.TRANSITION:

                if self.state_transition == TransitionState.EXIT_CROSS:
                    # 驶出十字路口处理
                    if self.locator.detect_black(grayscale_data):
                        self.api.move_forward(self.speed_move_in_white)
                    else:
                        self.state_transition = TransitionState.MOVE_FORWARD
                elif self.state_transition == TransitionState.MOVE_FORWARD:
                    if self.locator.detect_black(grayscale_data):
                        self.state_main = MainState.LOCATION
                        self.api.stop()
                    else:
                        self.api.move_forward(self.speed_move_in_white)
                elif self.state_transition == TransitionState.SPAN:
                    if self.state_span == SpanState.FORWARD:
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.EXIT_CROSS
                    elif self.state_span == SpanState.BACKWARD:
                        if not self.spanner_backward.in_progress:
                            self.spanner_backward.start()
                        if self.spanner_backward.complete():
                            self.state_main = MainState.RELOCATION
                            self.state_relocation = RelocationState.SHORT
                        else:
                            self.api.spin_left(self.speed_spin)
                    elif self.state_span == SpanState.LEFT:
                        if not self.spanner_left.in_progress:
                            self.spanner_left.start()
                        if self.spanner_left.complete():
                            self.state_main = MainState.RELOCATION
                            self.state_relocation = RelocationState.SHORT
                        else:
                            self.api.spin_left(self.speed_spin)
                    elif self.state_span == SpanState.RIGHT:
                        if not self.spanner_right.in_progress:
                            self.spanner_right.start()
                        if self.spanner_right.complete():
                            self.state_main = MainState.RELOCATION
                            self.state_relocation = RelocationState.SHORT
                        else:
                            self.api.spin_right(self.speed_spin)
            elif self.state_main == MainState.LOCATION:
                if self.locator.translate_to_center(grayscale_data):
                    # 中心对齐了

                    if self.locator.reach_target(grayscale_data, False):
                        self.api.stop()

                        self.__correct_direction()

                        if self.state_relocation == RelocationState.COMPLETE:

                            if self.state_target == TargetState.APRIL_TAG:

                                if self.count_cross_pass < 1:
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.SPAN
                                    self.state_span = SpanState.LEFT
                                else:
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.GESTURE:

                                if self.count_cross_pass < 1:
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.EXIT_CROSS
                                else:
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.YOLO:

                                if self.count_cross_pass < 1:
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.SPAN
                                    self.state_span = SpanState.RIGHT
                                else:
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.TURN_LEFT

                            elif self.state_target == TargetState.FACE:
                                self.state_main = MainState.RECOGNITION
                                self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.BACK_HOME:
                                self.state_main = MainState.LINE

                        else:
                            self.state_relocation = RelocationState.COMPLETE

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.EXIT_CROSS

                    else:
                        if self.locator.seeking_left(grayscale_data):
                            self.count_locate_spin_left += 1
                            self.api.spin_left(self.speed_locate_turn)
                        elif self.locator.seeking_right(grayscale_data):
                            self.count_locate_spin_right += 1
                            self.api.spin_right(self.speed_locate_turn)
                        else:
                            self.api.move_forward(int(self.speed_locate_move))

                else:
                    # 中心未对齐，但会出现都是 False 的情况，应当先解决次情况
                    if self.locator.move_straight(grayscale_data):
                        self.api.move_forward(self.speed_locate_move)

                    else:
                        if self.locator.move_left(grayscale_data):
                            self.api.move_left(self.speed_locate_move)
                        elif self.locator.move_right(grayscale_data):
                            self.api.move_right(self.speed_locate_move)
                        else:
                            pass

            elif self.state_main == MainState.RELOCATION:

                if self.state_relocation == RelocationState.SHORT:

                    if not self.timer_back_short.in_progress:
                        self.timer_back_short.start()

                    if self.timer_back_short.complete():
                        self.api.stop()
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.MOVE_FORWARD
                    else:
                        self.api.move_backward(self.speed_move_in_white)

                elif self.state_relocation == RelocationState.LONG:

                    if not self.timer_back_long.in_progress:
                        self.timer_back_long.start()

                    if self.timer_back_long.complete():
                        self.api.stop()
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.MOVE_FORWARD
                    else:
                        self.api.move_backward(self.speed_move_in_white)

            elif self.state_main == MainState.RECOGNITION:

                if self.state_recognition == RecognitionState.TURN_LEFT:

                    if not self.spanner_left.in_progress:
                        self.spanner_left.start()

                    if self.spanner_left.complete():
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.PREPARE
                    else:
                        self.api.spin_left(self.speed_spin)

                elif self.state_recognition == RecognitionState.PREPARE:

                    if self.count_stable < self.count_continuous_stable:
                        self.api.stop()
                        self.count_stable += 1
                        self.__clamp_arms()
                        continue

                    if self.state_target == TargetState.APRIL_TAG:
                        self.timer_arm_action.start()
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.EXECUTE

                    elif self.state_target == TargetState.GESTURE:
                        self.timer_arm_action.start()
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.EXECUTE

                    elif self.state_target == TargetState.YOLO:
                        preload_complete = self.api.preload_yolo_pool()
                        if preload_complete:
                            self.state_main = MainState.RECOGNITION
                            self.state_recognition = RecognitionState.AIM

                    elif self.state_target == TargetState.FACE:
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.AIM

                elif self.state_recognition == RecognitionState.AIM:

                    if self.state_target == TargetState.YOLO:
                        find_target, offset_x = self.api.detect_yolo(label=self.target_yolo)
                        if find_target:
                            self.__aim_target(offset_x)

                    elif self.state_target == TargetState.FACE:
                        find_target, offset_x = self.api.detect_face(label=self.target_face)
                        if find_target:
                            self.__aim_target(offset_x)

                elif self.state_recognition == RecognitionState.EXECUTE:

                    if self.state_target == TargetState.APRIL_TAG:

                        find_tag, tag_id, offset = self.api.detect_apriltag()

                        if self.timer_arm_action.complete():
                            self.__clamp_arms()

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.SPAN
                            self.state_span = SpanState.BACKWARD

                            self.state_target = TargetState.GESTURE
                            self.state_relocation = RelocationState.SHORT

                            self.api.close_tag_window()
                        else:
                            if find_tag:
                                print(f"找到 April Tag 动作：{tag_id}")
                                self.__do_arm_action(tag_id)

                    elif self.state_target == TargetState.GESTURE:

                        find_target, number = self.api.detect_gesture()

                        if self.timer_arm_action.complete():
                            self.__clamp_arms()

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.SPAN
                            self.state_span = SpanState.BACKWARD

                            self.state_target = TargetState.YOLO
                            self.state_relocation = RelocationState.SHORT

                            self.api.close_gesture_window()
                        else:
                            if find_target:
                                print(f"找到手势动作：{number}")
                                self.__do_arm_action(number)

                    elif self.state_target == TargetState.YOLO:
                        print(f"开始击打 YOLO 目标：{self.target_yolo}")
                        self.__hit_actions()

                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.SPAN
                        self.state_span = SpanState.RIGHT

                        self.state_target = TargetState.FACE
                        self.state_relocation = RelocationState.LONG

                        self.__reset_yolo()

                    elif self.state_target == TargetState.FACE:
                        print(f"开始击打人脸目标：{self.target_face}")
                        self.__hit_actions()

                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.SPAN
                        self.state_span = SpanState.RIGHT

                        self.state_target = TargetState.BACK_HOME
                        self.state_relocation = RelocationState.LONG

                        self.api.close_face_window()

            elif self.state_main == MainState.LINE:

                if self.locator.detect_black(grayscale_data):
                    self.__follow_line(line_center_offset)
                    self.count_out_line = 0
                else:
                    if self.count_out_line > self.count_max_out_line:
                        self.state_main = MainState.HOME
                    else:
                        self.count_out_line += 1

            elif self.state_main == MainState.HOME:

                if not self.timer_enter_home.in_progress:
                    self.timer_enter_home.start()

                if self.timer_enter_home.complete():
                    self.state_main = MainState.FINISH
                else:
                    self.api.move_forward(self.speed_follow_line)

            elif self.state_main == MainState.FINISH:
                self.api.stop()
                break

    def __aim_target(self, offset):
        if offset is not None:
            if offset >= self.target_center_offset:
                self.api.move_right(self.speed_aim)
            elif offset <= -self.target_center_offset:
                self.api.move_left(self.speed_aim)
            else:
                self.api.stop()
                self.state_recognition = RecognitionState.EXECUTE
        else:
            print(f"没有发现目标")
            self.api.stop()

    def __follow_line(self, offset):
        turn_rate = self.pid.compute(offset)
        self.api.move_rotation(speed=self.speed_follow_line, turn_rate=turn_rate)

    def __correct_direction(self):
        if self.count_locate_spin_left > self.count_max_locate_spin:
            print("右转矫正方向")
            self.api.spin_right(self.speed_spin, self.time_right_turn)
            time.sleep(self.time_right_turn / 1000)

        elif self.count_locate_spin_right > self.count_max_locate_spin:
            print("左转矫正方向")
            self.api.spin_left(self.speed_spin, self.time_left_turn)
            time.sleep(self.time_left_turn / 1000)

        else:
            print("不需要矫正方向")

        self.count_locate_spin_left = 0
        self.count_locate_spin_right = 0

    def __hit_actions(self):
        # 前进
        self.api.move_forward(speed=self.speed_hit_position, run_time=self.time_hit_position)
        time.sleep(self.time_hit_position / 1000)

        # 击打动作序列
        self.__pre_hit()
        time.sleep(self.time_arm_action / 1000)
        self.__hit()
        time.sleep(self.time_arm_action / 1000)
        self.__clamp_arms()

        # 后退
        self.api.move_backward(speed=self.speed_hit_position, run_time=self.time_hit_position)
        time.sleep(self.time_hit_position / 1000)

    def __do_arm_action(self, number):
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
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["clamp"]
        self.api.execute_arm_action(left_action, right_action)

    def __pre_hit(self):
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["pre_hit"]
        self.api.execute_arm_action(left_action, right_action)

    def __hit(self):
        left_action = self.left_arm_actions["clamp"]
        right_action = self.right_arm_actions["hit"]
        self.api.execute_arm_action(left_action, right_action)

    def __reset_yolo(self):
        self.api.close_yolo_window()
        self.api.reset_yolo_pool()


if __name__ == '__main__':
    controller = Controller()
    controller.run()
