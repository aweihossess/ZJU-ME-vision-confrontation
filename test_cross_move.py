from sdk.api import UpAPI
from sdk.logic_layer.cross_planner import CrossLocator
from sdk.logic_layer.time_meter import TimeMeter
from enum import Enum, auto

"""
在两个十字之间循环移动
"""


class StateMain(Enum):
    SET_OUT = auto()
    MOVE = auto()
    LOCATE = auto()
    SPAN = auto()
    RELOCATE = auto()


if __name__ == '__main__':
    # 参数设置
    locate_move_speed = 4  # 定位水平移动速度
    locate_turn_speed = 40  # 定位旋转速度
    move_speed = 12  # 水平移动速度
    span_speed = 50  # 旋转速度

    # 状态机
    state_main = StateMain.SET_OUT
    move_to_next_cross = False

    # 传感器和执行器
    api = UpAPI()

    # 逻辑处理器
    locator = CrossLocator()

    # 计时器
    spanner = TimeMeter(9000)
    relocate_back = TimeMeter(1000)
    short_stop = TimeMeter(500)

    while True:
        grayscale_data = api.get_grayscale_data()

        if state_main == StateMain.SET_OUT:
            if short_stop.complete():
                if locator.leave_cross(grayscale_data):
                    print("离开十字了， 开始移动...")
                    state_main = StateMain.MOVE
                else:
                    api.move_forward(move_speed)
            else:
                api.stop()

        elif state_main == StateMain.MOVE:
            if locator.detect_black(grayscale_data):
                print("找到十字了，开始定位...")
                state_main = StateMain.LOCATE
                move_to_next_cross = False
            else:
                api.move_forward(move_speed)

        elif state_main == StateMain.LOCATE:
            if locator.translate_to_center(grayscale_data):
                # 中心对齐了

                if locator.reach_target(grayscale_data):
                    print("我到家了!!!")
                    api.stop()

                    short_stop.start()
                    if move_to_next_cross:
                        print("准备出十字...")
                        state_main = StateMain.SET_OUT
                    else:
                        print("准备掉头...")
                        state_main = StateMain.SPAN

                else:
                    if locator.seeking_left(grayscale_data):
                        print("中心是黑色了，前进左转")
                        api.spin_left(locate_turn_speed)
                    elif locator.seeking_right(grayscale_data):
                        print("中心是黑色了，前进右转")
                        api.spin_right(locate_turn_speed)
                    else:
                        print("move")
                        api.move_forward(int(locate_move_speed))
            else:
                # 中心未对齐，但会出现都是 False 的情况，应当先解决次情况
                if locator.move_straight(grayscale_data):
                    print("All False")
                    api.move_forward(locate_move_speed)

                else:
                    if locator.move_left(grayscale_data):
                        print("中心未检测到黑色，左平移")
                        api.move_left(locate_move_speed)
                    elif locator.move_right(grayscale_data):
                        print("中心未检测到黑色，右平移")
                        api.move_right(locate_move_speed)
                    else:
                        print("无法判断了")
                        pass

        elif state_main == StateMain.SPAN:
            if short_stop.complete():
                if not spanner.in_progress:
                    spanner.start()
                else:
                    if spanner.complete():
                        state_main = StateMain.RELOCATE
                        short_stop.start()
                    else:
                        api.seeking(span_speed)

            else:
                api.stop()

        elif state_main == StateMain.RELOCATE:
            if short_stop.complete():
                if not relocate_back.in_progress:
                    relocate_back.start()
                else:
                    if relocate_back.complete():
                        state_main = StateMain.LOCATE
                        move_to_next_cross = True
                    else:
                        api.move_backward(move_speed)

            else:
                api.stop()

        else:
            print("未知状态")
