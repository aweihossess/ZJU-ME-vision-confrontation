from sdk.logic_layer.cross_planner import CrossLocator
from sdk.api import UpAPI
from enum import Enum, auto


class State(Enum):
    LINE = auto()
    LOCATION = auto()


if __name__ == '__main__':
    # 参数设置
    locate_speed = 4
    line_follower_speed = 8
    turn_speed = 20

    # 状态机
    state = State.LINE

    # 传感器和执行器
    api = UpAPI()

    # 定位逻辑
    locator = CrossLocator()

    # 超时次数（超过阈值后，开始巡线）
    all_false_count = 0

    while True:
        grayscale_data = api.get_grayscale_data()

        if state == State.LINE:
            print("巡线前进")
            if locator.detect_black(grayscale_data):
                print("检测得到黑色了，进入定位环节")
                state = State.LOCATION
                api.stop()
            else:
                api.move_forward(line_follower_speed)

        elif state == State.LOCATION:
            if locator.translate_to_center(grayscale_data):
                # 中心对齐了

                if locator.reach_target(grayscale_data):
                    print("我到家了")
                    api.stop()

                else:
                    if locator.translate_left(grayscale_data):
                        print("中心是黑色了，左平移")
                        api.move_left(locate_speed)
                    elif locator.translate_right(grayscale_data):
                        print("中心是黑色了，右平移")
                        api.move_right(locate_speed)
                    elif locator.seeking_left(grayscale_data):
                        print("中心是黑色了，前进左转")
                        api.spin_left(turn_speed)
                    elif locator.seeking_right(grayscale_data):
                        print("中心是黑色了，前进右转")
                        api.spin_right(turn_speed)
                    else:
                        print("move")
                        api.move_forward(locate_speed)
            else:

                # 中心未对齐，但会出现都是 False 的情况，应当先解决次情况
                if locator.move_straight(grayscale_data):
                    print("all false")
                    api.move_forward(locate_speed)
                else:

                    if any(grayscale_data[:3]):
                        print("中心未检测到黑色，左平移")
                        api.move_left(locate_speed)
                    elif any(grayscale_data[4:]):
                        print("中心未检测到黑色，右平移")
                        api.move_right(locate_speed)
                    else:
                        print("无法判断了")
                        pass

        else:
            print("unknown state")
