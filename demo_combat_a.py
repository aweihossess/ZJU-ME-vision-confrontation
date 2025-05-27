# 仿人机器人格斗 A

from sdk.data_layer.arm import arm_action_factory as arm_action
from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.time_meter import TimeMeter
from enum import Enum, auto
import time


class MainState(Enum):
    IDLE = auto()
    LINE = auto()
    TRANSITION = auto()
    LOCATION = auto()
    RELOCATION = auto()
    RECOGNITION = auto()
    FINISH = auto()


class TransitionState(Enum):
    ENTER_CROSS = auto()
    EXIT_CROSS = auto()
    TURN_RIGHT = auto()


class RelocationState(Enum):
    YES = auto()
    NO = auto()


class ExitCrossState(Enum):
    BLACK = auto()
    WHITE = auto()


class RecognitionState(Enum):
    PREPARE = auto()
    AIM = auto()
    EXECUTE = auto()


class AreaState(Enum):
    ACTION = auto()  # 做动作区域
    BEAT = auto()  # 敲锣区域


class Controller:
    def __init__(self):
        # 参数设置
        self.grayscale_threshold = 3060  # 灰度传感器黑色检测阈值

        self.speed_follow_line = 16  # 巡线前进移动速度
        self.turn_ratio = 8  # 巡线矫正移动速度
        self.speed_locate_move = 4  # 定位移动速度
        self.speed_locate_turn = 20  # 定位旋转速度
        self.speed_span = 100  # 旋转速度
        self.speed_aim = 4  # 瞄准移动速度
        self.speed_hit_position = 10  # 前进到能够击打武器的位置的速度

        self.time_init = 1000  # 初始化时间，单位毫秒
        self.time_span = 1900  # 旋转时间，单位毫秒
        self.time_arm_execute = 2000  # 手臂动作执行时间，单位毫秒
        self.time_relocation_back = 800  # 重定位后退时间，单位毫秒
        self.time_hit_position = 1200  # 前进到能够击打武器的位置的时间，单位毫秒

        # YOLO 目标参数
        self.yolo_model = YoloModel.WEAPON  # YOLO 识别模式
        self.yolo_target = ("knife", "sword", "hammer")  # YOLO 目标标签
        self.yolo_edge = 35  # YOLO 目标边界

        # April Tag 识别
        self.tag_actions = {
            "left_up": 1,
            "right_up": 3,
            "both_up": 5
        }

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
        self.state_translation = TransitionState.EXIT_CROSS
        self.state_relocation = RelocationState.NO
        self.state_exit_cross = ExitCrossState.BLACK
        self.state_recognition = RecognitionState.PREPARE
        self.state_area = AreaState.ACTION

        # 传感器和执行器
        self.api = UpAPI(yolo_model=self.yolo_model, grayscale_threshold=self.grayscale_threshold)

        # 逻辑处理器
        self.locator = CrossLocator()

        # 计时器
        self.initializer = TimeMeter(self.time_init)  # 初始化
        self.spanner = TimeMeter(self.time_span)  # 旋转
        self.timer_april_tag = TimeMeter(self.time_arm_execute)  # April Tag 识别时间
        self.timer_back = TimeMeter(self.time_relocation_back)  # 重定位后退
        self.timer_hit_position = TimeMeter(self.time_hit_position)  # 前进到能够击打武器的位置

        # 准备从巡线进入黑色十字
        self.leave_line_count = 0  # 离开巡线
        self.continuous_leave_line_count = 10  # 当 white_count 大于此值，认为已经离开巡线

        # 相机稳定
        self.stable_count = 0  # 相机稳定计数器
        self.continuous_stable_count = 15  # 相机连续稳定阈值

        # 已经定位十字的次数
        self.cross_pass_count = 0  # 十字定位次数计数器
        self.change_area_count = 4  # 定位 4 次后进入敲锣区域

        # 武器打击次数
        self.weapon_hit_count = 0
        self.max_weapon_hit_count = len(self.yolo_target) - 1

        # 没有目标的次数
        self.no_target_count = 0
        self.max_no_target_count = 5

    def run(self):
        while True:
            # 传感器数据
            grayscale_data = self.api.get_grayscale_data()

            # 数据处理
            line_center_offset = self.api.follow_line()
            find_apriltag, tag_id, tag_offset = self.api.detect_apriltag()

            # 状态机
            if self.state_main == MainState.IDLE:
                print("初始化...")
                if self.initializer.complete():
                    self.__exit_cross()
                else:
                    self.__clamp_arms()

            elif self.state_main == MainState.LINE:
                black = self.locator.detect_black(grayscale_data)
                print(f"巡线中 {black}, {grayscale_data}")
                if black:
                    self.__follow_line(line_center_offset, self.speed_follow_line)
                    self.leave_line_count = 0
                else:
                    if self.leave_line_count >= self.continuous_leave_line_count:
                        self.__prepare_to_enter_cross()
                    else:
                        self.leave_line_count += 1

            elif self.state_main == MainState.TRANSITION:

                if self.state_translation == TransitionState.ENTER_CROSS:
                    if self.locator.detect_black(grayscale_data):
                        print("检测到黑色，进入定位环节")
                        self.__prepare_to_locate()

                    else:
                        print("准备进入黑色十字，前进中")
                        self.api.move_forward(self.speed_locate_move)

                elif self.state_translation == TransitionState.EXIT_CROSS:
                    if self.state_exit_cross == ExitCrossState.BLACK:
                        if self.locator.detect_black(grayscale_data):
                            print("在黑色十字中前进")
                            self.__follow_line(line_center_offset, self.speed_locate_move)

                        else:
                            print("进入白色区域")
                            self.state_exit_cross = ExitCrossState.WHITE

                    elif self.state_exit_cross == ExitCrossState.WHITE:
                        if self.locator.detect_black(grayscale_data):
                            print("检测到黑色，准备巡线")
                            self.state_main = MainState.LINE

                        else:
                            print("在白色区域中前进")
                            self.api.move_forward(self.speed_locate_move)

                elif self.state_translation == TransitionState.TURN_RIGHT:
                    if not self.spanner.in_progress:
                        self.spanner.start()

                    else:
                        if self.spanner.complete():
                            self.__prepare_to_relocate()
                        else:
                            self.api.spin_right(self.speed_span)

            elif self.state_main == MainState.LOCATION:
                if self.locator.translate_to_center(grayscale_data):
                    # 中心对齐了

                    if self.locator.reach_target(grayscale_data):
                        print("我到家了!!!")
                        self.api.stop()

                        if self.state_relocation == RelocationState.YES:
                            self.__exit_cross()

                        elif self.state_relocation == RelocationState.NO:
                            self.__count_passed_cross()
                            self.__prepare_to_recognize()

                    else:
                        if self.locator.seeking_left(grayscale_data):
                            print("中心是黑色了，前进左转")
                            self.api.spin_left(self.speed_locate_turn)
                        elif self.locator.seeking_right(grayscale_data):
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
                if self.timer_back.complete():
                    self.api.stop()
                    self.__prepare_to_locate()

                else:
                    self.api.move_backward(self.speed_follow_line)

            elif self.state_main == MainState.RECOGNITION:

                if self.state_recognition == RecognitionState.PREPARE:

                    # 等待相机稳定
                    if self.stable_count < self.continuous_stable_count:
                        print("等待相机稳定")
                        self.api.stop()
                        self.stable_count += 1
                        self.__clamp_arms()
                        continue

                    if self.state_area == AreaState.ACTION:
                        print("演习区域，准备识别 April Tag")
                        self.timer_april_tag.start()
                        self.state_recognition = RecognitionState.EXECUTE

                    elif self.state_area == AreaState.BEAT:
                        print("武器选择区域，预加载图像")
                        preload_complete = self.api.preload_yolo_pool()
                        if preload_complete:
                            print("预加载图像完成，准备瞄准")
                            self.__prepare_to_aim()

                    else:
                        print(f"未知区域状态: {self.state_area}")

                elif self.state_recognition == RecognitionState.AIM:
                    if self.state_area == AreaState.BEAT:
                        print("预加载图像完成，准备瞄准")

                        if self.__select_weapon_complete():
                            self.state_main = MainState.FINISH
                            continue

                        target_name = self.yolo_target[self.weapon_hit_count]
                        find_target, offset_x = self.api.detect_yolo(label=target_name)

                        if find_target:
                            print("找到目标，开始瞄准")

                            self.no_target_count = 0
                            self.__aim_target(offset_x)

                        else:
                            if self.no_target_count > self.max_no_target_count:
                                print("没有找到目标，向右移动")

                                self.api.move_right(self.speed_locate_move)

                            else:
                                self.no_target_count += 1

                    else:
                        print(f"瞄准区域状态: {self.state_area}")

                elif self.state_recognition == RecognitionState.EXECUTE:
                    if self.state_area == AreaState.ACTION:

                        if self.timer_april_tag.complete():
                            self.__prepare_to_turn_right()
                            self.__clamp_arms()

                        else:
                            if find_apriltag:
                                self.__move_arm_by_number(tag_id)

                    elif self.state_area == AreaState.BEAT:
                        self.__hit_weapon_actions()
                        self.weapon_hit_count += 1
                        self.state_recognition = RecognitionState.AIM

                    else:
                        print(f"未知区域状态: {self.state_area}")

            elif self.state_main == MainState.FINISH:
                print("任务完成")
                self.api.stop()
                break

    def __exit_cross(self):
        self.state_main = MainState.TRANSITION
        self.state_translation = TransitionState.EXIT_CROSS
        self.state_exit_cross = ExitCrossState.BLACK
        self.state_relocation = RelocationState.NO

    def __follow_line(self, offset, move_speed):
        self.api.move_rotation(speed=move_speed, turn_rate=int(offset * self.turn_ratio))

    def __prepare_to_enter_cross(self):
        self.state_main = MainState.TRANSITION
        self.state_translation = TransitionState.ENTER_CROSS

    def __prepare_to_locate(self):
        self.state_main = MainState.LOCATION
        self.api.stop()

    def __prepare_to_relocate(self):
        self.state_main = MainState.RELOCATION
        self.state_relocation = RelocationState.YES
        self.timer_back.start()
        self.api.stop()

    def __prepare_to_recognize(self):
        self.state_main = MainState.RECOGNITION
        self.state_recognition = RecognitionState.PREPARE

    def __prepare_to_turn_right(self):
        self.state_main = MainState.TRANSITION
        self.state_translation = TransitionState.TURN_RIGHT

    def __prepare_to_aim(self):
        if not self.timer_hit_position.in_progress:
            self.timer_hit_position.start()

        if self.timer_hit_position.complete():
            self.state_recognition = RecognitionState.AIM
        else:
            self.api.move_forward(self.speed_hit_position)

    def __count_passed_cross(self):
        self.cross_pass_count += 1
        if self.cross_pass_count >= self.change_area_count:
            self.state_area = AreaState.BEAT

    def __aim_target(self, offset):
        if offset is not None:
            if offset >= self.yolo_edge:
                self.api.move_right(self.speed_aim)
            elif offset <= -self.yolo_edge:
                self.api.move_left(self.speed_aim)
            else:
                print(f"瞄准武器：{self.yolo_target[self.weapon_hit_count]}")
                self.api.stop()
                self.state_recognition = RecognitionState.EXECUTE

        else:
            print(f"没有发现目标")
            self.api.move_right(self.speed_aim)

    def __select_weapon_complete(self):
        return self.weapon_hit_count > self.max_weapon_hit_count

    def __hit_weapon_actions(self):
        print(f"击打武器：{self.yolo_target[self.weapon_hit_count]}")
        self.__pre_hit()
        time.sleep(self.time_arm_execute / 1000)
        self.__hit()
        time.sleep(self.time_arm_execute / 1000)
        self.__clamp_arms()

    def __move_arm_by_number(self, number):
        if number == self.tag_actions["left_up"]:
            # 举左手
            left_action = self.left_arm_actions["up"]
            right_action = self.right_arm_actions["clamp"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == self.tag_actions["right_up"]:
            # 举右手
            left_action = self.left_arm_actions["clamp"]
            right_action = self.right_arm_actions["up"]
            self.api.execute_arm_action(left_action, right_action)

        elif number == self.tag_actions["both_up"]:
            # 举双手
            left_action = self.left_arm_actions["up"]
            right_action = self.right_arm_actions["up"]
            self.api.execute_arm_action(left_action, right_action)

        else:
            print(f"未指定动作: {number}")

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


if __name__ == '__main__':
    controller = Controller()
    controller.run()
