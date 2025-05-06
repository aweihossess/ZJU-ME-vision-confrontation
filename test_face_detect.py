from sdk.api import UpAPI
from sdk.data_layer.arm import arm_action_factory as arm_data
import time


def get_time_ms():
    return int(time.time() * 1000)


class State:
    IDLE, STOP, LEFT, RIGHT, FINISH, HIT = range(6)
    PRE_BEAT, BEAT, RESET = range(3)

    def __init__(self):
        self.state_main = self.IDLE
        self.state_hit = self.RESET

        self.distance = 30

        self.last_hit_time = get_time_ms()
        self.hit_time = 2000

        self.start_init_time = get_time_ms()
        self.init_time = 5000

    def update_face(self, find_faces, offset_x):
        if self.state_main == self.IDLE:
            if get_time_ms() - self.start_init_time >= self.init_time:
                self.state_main = self.STOP
            else:
                return

        if find_faces:
            if offset_x is not None:
                if offset_x >= self.distance:
                    self.state_main = self.RIGHT
                elif offset_x <= -self.distance:
                    self.state_main = self.LEFT
                else:
                    self.state_main = self.HIT

                    if self.state_hit == self.RESET:
                        if get_time_ms() - self.last_hit_time >= self.hit_time:
                            self.last_hit_time = get_time_ms()
                            self.state_hit = self.PRE_BEAT

                    elif self.state_hit == self.PRE_BEAT:
                        if get_time_ms() - self.last_hit_time >= self.hit_time:
                            self.last_hit_time = get_time_ms()
                            self.state_hit = self.BEAT

                    elif self.state_hit == self.BEAT:
                        if get_time_ms() - self.last_hit_time >= self.hit_time:
                            self.last_hit_time = get_time_ms()
                            self.state_main = self.FINISH

            else:
                print(f"没有发现目标")
                self.state_main = self.STOP

        else:
            print("未检测到人脸")
            self.state_main = self.STOP


if __name__ == '__main__':
    interval = 0.05

    target_label = "t_0"

    arm_reset = {
        "left": arm_data.left_arm_clamp(),
        "right": arm_data.right_arm_clamp()
    }
    arm_pre_hit = {
        "left": arm_data.left_arm_clamp(),
        "right": arm_data.right_arm_prepare_beat()
    }
    arm_hit = {
        "left": arm_data.left_arm_clamp(),
        "right": arm_data.right_arm_beat()
    }

    api = UpAPI()

    state = State()

    while True:
        # 获取数据
        find_face, offset_x = api.detect_face(label=target_label)
        # print(offset_x)

        # 更新状态
        state.update_face(find_face, offset_x)

        # 执行
        if state.state_main == state.IDLE:
            api.execute_arm_action(arm_reset["left"], arm_reset["right"])

        elif state.state_main == state.STOP:
            api.stop()

        elif state.state_main == state.FINISH:
            print("finished")
            break

        elif state.state_main == state.LEFT:
            api.move_left(10)

        elif state.state_main == state.RIGHT:
            api.move_right(10)

        elif state.state_main == state.HIT:
            api.stop()

            if state.state_hit == state.RESET:
                api.execute_arm_action(arm_reset["left"], arm_reset["right"])

            elif state.state_hit == state.PRE_BEAT:
                api.execute_arm_action(arm_pre_hit["left"], arm_pre_hit["right"])

            elif state.state_hit == state.BEAT:
                api.execute_arm_action(arm_hit["left"], arm_hit["right"])

    time.sleep(interval)