from sdk.api import UpAPI
from sdk.model import YoloModel
from sdk.data_layer.arm import arm_action_factory as arm_data
import time


def get_time_ms():
    return int(time.time() * 1000)


class State:
    # 状态机各个子状态 赋值
    IDLE, STOP, LEFT, RIGHT, FINISH, HIT = range(6)
    PRE_BEAT, BEAT, RESET = range(3)

    label = {
        "vehicle": "tank",
        "weapon": "hammer"  # "hammer", "sword", "knife"
    }

    def __init__(self, model):
        self.label = self.__get_yolo_label(model)

        self.state_main = self.IDLE
        self.state_hit = self.RESET

        self.distance = 30

        self.last_hit_time = get_time_ms()
        self.hit_time = 2000

        self.start_init_time = get_time_ms()
        self.init_time = 5000

    def update_target(self, find_target, offset_x):
        if self.state_main == self.IDLE:
            if get_time_ms() - self.start_init_time >= self.init_time:
                self.state_main = self.STOP
            else:
                return

        elif self.state_main == self.HIT:
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

            return


        if find_target:
            if offset_x is not None:
                if offset_x >= self.distance:
                    self.state_main = self.RIGHT
                elif offset_x <= -self.distance:
                    self.state_main = self.LEFT
                else:
                    self.state_main = self.HIT

            else:
                print(f"没有发现目标")
                self.state_main = self.STOP

        else:
            print("未检测到目标")
            self.state_main = self.STOP

    def __get_yolo_label(self, model):
        if model == YoloModel.VEHICLE:
            return self.label["vehicle"]

        elif model == YoloModel.WEAPON:
            return self.label["weapon"]

        else:
            raise RuntimeError(f"unknown model: {model}")


if __name__ == '__main__':
    model = YoloModel.WEAPON
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

    api = UpAPI(yolo_model=model)

    state = State(model)

    while True:
        preload_complete = api.preload_yolo_pool()
        if preload_complete:
            find_target, offset_x = api.detect_yolo()
            # print(offset_x)

            # 更新状态
            state.update_target(find_target, offset_x)

            # 执行
            if state.state_main == state.IDLE:
                api.execute_arm_action(arm_reset["left"], arm_reset["right"])

            elif state.state_main == state.STOP:
                api.stop()

            elif state.state_main == state.FINISH:
                print("finished")

            elif state.state_main == state.LEFT:
                api.move_left(50)

            elif state.state_main == state.RIGHT:
                api.move_right(50)

            elif state.state_main == state.HIT:
                api.stop()

                if state.state_hit == state.RESET:
                    api.execute_arm_action(arm_reset["left"], arm_reset["right"])

                elif state.state_hit == state.PRE_BEAT:
                    api.execute_arm_action(arm_pre_hit["left"], arm_pre_hit["right"])

                elif state.state_hit == state.BEAT:
                    api.execute_arm_action(arm_hit["left"], arm_hit["right"])

        # else:
        #     print("filling pool")