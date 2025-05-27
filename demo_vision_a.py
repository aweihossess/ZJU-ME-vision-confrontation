# 机器人视觉竞技 A

from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.pid import PIDController
from sdk.logic_layer.time_meter import TimeMeter
from enum import Enum, auto
import time


class MainState(Enum):
    """
    一级状态机：主状态（行动状态）
    """
    # 初始化状态，程序启动后首先进入该状态，进行一些初始化操作，当初始化完成后会切换到其他状态
    IDLE = auto()
    # 转场状态，用于在不同行动区域之间进行过渡，例如驶出十字、在白色区域行驶、旋转等操作
    TRANSITION = auto()
    # 定位状态，机器人会尝试将自身位置调整到目标中心，当中心对齐且到达目标位置时，会切换到其他状态
    LOCATION = auto()
    # 重定位状态，机器人会进行短距离或长距离的后退操作，为后续的定位或识别任务做准备
    RELOCATION = auto()
    # 识别状态，机器人会进行 April Tag 识别、手势识别、YOLO 识别或人脸识别等操作
    RECOGNITION = auto()
    # 巡线状态，机器人会沿着黑色线条进行移动，当离开线条达到一定次数后，会切换到其他状态
    LINE = auto()
    # 回家状态，机器人在巡线结束后会进入该状态，朝着指定的家的位置移动
    HOME = auto()
    # 完成状态，机器人完成所有任务后会进入该状态，停止所有动作
    FINISH = auto()

class TargetState(Enum):
    """
    一级状态机：目标状态（行动区域）
    """
    APRIL_TAG = auto()
    GESTURE = auto()
    YOLO = auto()
    FACE = auto()
    BACK_HOME = auto()


class TransitionState(Enum):
    """
    二级状态机：转场状态
    """
    EXIT_CROSS = auto()  # 驶出十字
    MOVE_FORWARD = auto()  # 在白色区域中行驶
    SPAN = auto()  # 旋转


class RelocationState(Enum):
    """
    二级状态机：重定位状态
    """
    LONG = auto()  # 长距离后退，YOLO 和人脸检测区域使用
    SHORT = auto()  # 短距离后退，其余区域使用
    COMPLETE = auto()  # 重定位完成


class RecognitionState(Enum):
    """
    二级状态机：识别状态
    """
    TURN_LEFT = auto()  # 仅有 YOLO 检测之前使用
    PREPARE = auto()
    AIM = auto()
    EXECUTE = auto()


class SpanState(Enum):
    """
    三级状态机：转场的旋转状态
    """
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()


class Controller:
    def __init__(self):
        # 参数设置
        self.grayscale_threshold = 1600  # 灰度传感器检测阈值

        self.speed_follow_line = 10  # 巡线前进移动速度
        self.speed_move_in_white = 8  # 在白色区域前进移动速度
        self.speed_hit_position = 8  # 进入退出击打位置的速度
        self.speed_locate_move = 4  # 定位移动速度
        self.speed_locate_turn = 20  # 定位旋转速度
        self.speed_spin = 50  # 自旋速度
        self.speed_aim = 4  # 瞄准移动速度

        self.time_init = 1000  # 初始化时间，单位毫秒
        self.time_left_turn = 3800  # 向左转时间，单位毫秒
        self.time_right_turn = 3360  # 向右转时间，单位毫秒
        self.time_backward_turn = 7160  # 向后转时间，单位毫秒
        self.time_arm_action = 2500  # 手臂做动作时间，单位毫秒
        self.time_enter_home = 1500  # 巡线结束到停车之间的时间，单位毫秒
        self.time_hit_position = 1000  # 进入退出击打状态的移动时间，单位毫秒
        self.time_back_short = 800  # 重定位短距离后退，单位毫秒
        self.time_back_long = 1900  # 重定位长距离后退，单位毫秒

        self.k_p = 15  # 巡线比例参数
        self.k_i = 0.2  # 巡线积分参数
        self.k_d = 1.0  # 巡线微分参数

        self.target_face = "t_0"  # 人脸识别标签
        self.target_yolo = "tank"  # 车辆识别标签
        self.target_id = 1  # April Tag 识别 ID  （实际是 1 号）
        self.target_number_right = 0  # 手势识别数字，举右手
        self.target_number_both = 5  # 手势识别数字，举右手

        self.target_center_offset = 35  # 目标中心与屏幕中心偏移量，单位像素

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

        # 状态机
        self.state_main = MainState.IDLE
        self.state_target = TargetState.APRIL_TAG
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
        self.count_cross_pass = 0  # 十字定位次数计数器

        # 结束巡线的次数
        self.count_out_line = 0
        self.count_max_out_line = 5

        # 定位旋转超出预计的次数
        self.count_locate_spin_left = 0
        self.count_locate_spin_right = 0
        self.count_max_locate_spin = 25

    def run(self):

        while True:

            # 传感器数据
            grayscale_data = self.api.get_grayscale_data()

            # 数据处理
            line_center_offset = self.api.follow_line()

            # 状态机
            if self.state_main == MainState.IDLE:

                if self.initializer.complete():
                    print("初始化完成")
                    self.state_main = MainState.TRANSITION
                    self.state_transition = TransitionState.EXIT_CROSS
                else:
                    print("初始化中...")
                    self.__clamp_arms()

            elif self.state_main == MainState.TRANSITION:

                if self.state_transition == TransitionState.EXIT_CROSS:

                    if self.locator.detect_black(grayscale_data):
                        print("检测到黑色")
                        self.api.move_forward(self.speed_move_in_white)
                    else:
                        print("准备进入白色区域前进")
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.MOVE_FORWARD

                elif self.state_transition == TransitionState.MOVE_FORWARD:

                    if self.locator.detect_black(grayscale_data):
                        print("检测到黑色，进入定位环节")
                        self.state_main = MainState.LOCATION
                        self.api.stop()
                    else:
                        print("白色区域中前进")
                        self.api.move_forward(self.speed_move_in_white)

                elif self.state_transition == TransitionState.SPAN:

                    if self.state_span == SpanState.FORWARD:
                        print("向前进，准备驶出十字")
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.EXIT_CROSS

                    elif self.state_span == SpanState.BACKWARD:

                        if not self.spanner_backward.in_progress:
                            self.spanner_backward.start()

                        if self.spanner_backward.complete():
                            print("向后转完成，准备进入短距离重定位")
                            self.state_main = MainState.RELOCATION
                            self.state_relocation = RelocationState.SHORT

                        else:
                            print("向后转中")
                            self.api.spin_left(self.speed_spin)

                    elif self.state_span == SpanState.LEFT:

                        if not self.spanner_left.in_progress:
                            self.spanner_left.start()

                        if self.spanner_left.complete():
                            print("向左转完成，准备进入短距离重定位")
                            self.state_main = MainState.RELOCATION
                            self.state_relocation = RelocationState.SHORT

                        else:
                            print("向左转中")
                            self.api.spin_left(self.speed_spin)

                    elif self.state_span == SpanState.RIGHT:

                        if not self.spanner_right.in_progress:
                            self.spanner_right.start()

                        if self.spanner_right.complete():

                            if self.state_target == TargetState.YOLO:
                                print("向右转完成，准备进入短距离重定位")
                                self.state_main = MainState.RELOCATION
                                self.state_relocation = RelocationState.SHORT

                            elif (self.state_target == TargetState.BACK_HOME or
                                  self.state_target == TargetState.FACE):
                                print("向右转完成，准备进入长距离重定位")
                                self.state_main = MainState.RELOCATION
                                self.state_relocation = RelocationState.LONG

                        else:
                            print("向右转中")
                            self.api.spin_right(self.speed_spin)

            elif self.state_main == MainState.LOCATION:

                if self.locator.translate_to_center(grayscale_data):
                    # 中心对齐了

                    if self.locator.reach_target(grayscale_data, False):
                        print("定位成功!!!")
                        self.api.stop()

                        self.__correct_direction()

                        if self.state_relocation == RelocationState.COMPLETE:

                            if self.state_target == TargetState.APRIL_TAG:

                                if self.count_cross_pass < 1:
                                    print("到达 April Tag 识别前的一个十字，准备左转")
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.SPAN
                                    self.state_span = SpanState.LEFT
                                else:
                                    print("到达 April Tag 识别十字，准备进入识别程序")
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.GESTURE:

                                if self.count_cross_pass < 1:
                                    print("到达手势识别前的一个十字，准备继续前进")
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.EXIT_CROSS
                                else:
                                    print("到达手势识别十字，准备进入识别程序")
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.YOLO:

                                if self.count_cross_pass < 1:
                                    print("到达 YOLO 识别前的一个十字，准备右转")
                                    self.count_cross_pass += 1

                                    self.state_main = MainState.TRANSITION
                                    self.state_transition = TransitionState.SPAN
                                    self.state_span = SpanState.RIGHT
                                else:
                                    print("到达 YOLO 识别十字，准备先左转，再进入识别程序")
                                    self.count_cross_pass = 0

                                    self.state_main = MainState.RECOGNITION
                                    self.state_recognition = RecognitionState.TURN_LEFT

                            elif self.state_target == TargetState.FACE:
                                print("到达人脸识别十字，进入识别程序")
                                self.state_main = MainState.RECOGNITION
                                self.state_recognition = RecognitionState.PREPARE

                            elif self.state_target == TargetState.BACK_HOME:
                                print("准备进入巡线")
                                self.state_main = MainState.LINE

                        else:
                            print("重定位完成，准备向前驶出十字")
                            self.state_relocation = RelocationState.COMPLETE

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.EXIT_CROSS

                    else:
                        if self.locator.seeking_left(grayscale_data):
                            print("中心是黑色了，前进左转")
                            self.count_locate_spin_left += 1
                            self.api.spin_left(self.speed_locate_turn)
                        elif self.locator.seeking_right(grayscale_data):
                            self.count_locate_spin_right += 1
                            print("中心是黑色了，前进右转")
                            self.api.spin_right(self.speed_locate_turn)
                        else:
                            print("中心是黑色，继续前进")
                            self.api.move_forward(int(self.speed_locate_move))

                else:
                    # 中心未对齐，但会出现都是 False 的情况，应当先解决次情况
                    if self.locator.move_straight(grayscale_data):
                        print("All False")
                        self.api.move_forward(self.speed_locate_move)

                    else:
                        if self.locator.move_left(grayscale_data):
                            print("中心未检测到黑色，左平移")
                            self.api.move_left(self.speed_locate_move)
                        elif self.locator.move_right(grayscale_data):
                            print("中心未检测到黑色，右平移")
                            self.api.move_right(self.speed_locate_move)
                        else:
                            print("无法判断了")
                            pass

            elif self.state_main == MainState.RELOCATION:

                if self.state_relocation == RelocationState.SHORT:

                    if not self.timer_back_short.in_progress:
                        self.timer_back_short.start()

                    if self.timer_back_short.complete():
                        print("短距离后退完成，准备在白色区域中前进")
                        self.api.stop()
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.MOVE_FORWARD
                    else:
                        print("短距离后退中")
                        self.api.move_backward(self.speed_move_in_white)

                elif self.state_relocation == RelocationState.LONG:

                    if not self.timer_back_long.in_progress:
                        self.timer_back_long.start()

                    if self.timer_back_long.complete():
                        print("长距离后退完成，准备在白色区域中前进")
                        self.api.stop()
                        self.state_main = MainState.TRANSITION
                        self.state_transition = TransitionState.MOVE_FORWARD
                    else:
                        print("长距离后退中")
                        self.api.move_backward(self.speed_move_in_white)

            elif self.state_main == MainState.RECOGNITION:

                if self.state_recognition == RecognitionState.TURN_LEFT:

                    if not self.spanner_left.in_progress:
                        self.spanner_left.start()

                    if self.spanner_left.complete():
                        print("YOLO 识别前，左转完成")
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.PREPARE
                    else:
                        print("YOLO 识别前，左转中")
                        self.api.spin_left(self.speed_spin)

                elif self.state_recognition == RecognitionState.PREPARE:

                    if self.count_stable < self.count_continuous_stable:
                        print("等待相机稳定")
                        self.api.stop()
                        self.count_stable += 1
                        self.__clamp_arms()
                        continue

                    if self.state_target == TargetState.APRIL_TAG:
                        print("演习区域，准备识别 April Tag")
                        self.timer_arm_action.start()
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.EXECUTE

                    elif self.state_target == TargetState.GESTURE:
                        print("演习区域，准备识别手势图像")
                        self.timer_arm_action.start()
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.EXECUTE

                    elif self.state_target == TargetState.YOLO:
                        print("YOLO 识别区域，预加载图像中")
                        preload_complete = self.api.preload_yolo_pool()
                        if preload_complete:
                            print("预加载图像完成，准备瞄准")
                            self.state_main = MainState.RECOGNITION
                            self.state_recognition = RecognitionState.AIM

                    elif self.state_target == TargetState.FACE:
                        print("人脸识别区域，准备瞄准")
                        self.state_main = MainState.RECOGNITION
                        self.state_recognition = RecognitionState.AIM

                elif self.state_recognition == RecognitionState.AIM:

                    if self.state_target == TargetState.YOLO:
                        print("YOLO 预加载图像完成，准备瞄准")
                        find_target, offset_x = self.api.detect_yolo(label=self.target_yolo)
                        if find_target:
                            self.__aim_target(offset_x)

                    elif self.state_target == TargetState.FACE:
                        print("人脸识别区域，准备瞄准")
                        find_target, offset_x = self.api.detect_face(label=self.target_face)
                        if find_target:
                            self.__aim_target(offset_x)

                elif self.state_recognition == RecognitionState.EXECUTE:

                    if self.state_target == TargetState.APRIL_TAG:

                        find_tag, tag_id, offset = self.api.detect_apriltag()

                        if self.timer_arm_action.complete():
                            print("April Tag 动作完成")
                            self.__clamp_arms()

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.SPAN
                            self.state_span = SpanState.BACKWARD

                            self.state_target = TargetState.GESTURE
                            self.state_relocation = RelocationState.SHORT

                            self.api.close_tag_window()
                        else:
                            print("做 April Tag 动作中")
                            if find_tag:
                                print(f"找到 April Tag 动作：{tag_id}")
                                self.__do_arm_action(tag_id)

                    elif self.state_target == TargetState.GESTURE:

                        find_target, number = self.api.detect_gesture()

                        if self.timer_arm_action.complete():
                            print("手势动作完成")
                            self.__clamp_arms()

                            self.state_main = MainState.TRANSITION
                            self.state_transition = TransitionState.SPAN
                            self.state_span = SpanState.BACKWARD

                            self.state_target = TargetState.YOLO
                            self.state_relocation = RelocationState.SHORT

                            self.api.close_gesture_window()
                        else:
                            print("做手势动作中")
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
                    print("巡线中")
                    self.__follow_line(line_center_offset)
                    self.count_out_line = 0
                else:
                    if self.count_out_line > self.count_max_out_line:
                        print("巡线完成")
                        self.state_main = MainState.HOME
                    else:
                        self.count_out_line += 1

            elif self.state_main == MainState.HOME:

                if not self.timer_enter_home.in_progress:
                    self.timer_enter_home.start()

                if self.timer_enter_home.complete():
                    print("回家，停车")
                    self.state_main = MainState.FINISH
                else:
                    print("回家中")
                    self.api.move_forward(self.speed_follow_line)

            elif self.state_main == MainState.FINISH:
                print("任务完成")
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
