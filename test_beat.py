from sdk.api import UpAPI
from sdk.data_layer.arm import arm_action_factory as data_arm
import time


def get_time_ms():
    return int(time.time() * 1000)


class Controller:
    HOVER, LEFT_DOWN_RIGHT_UP = range(2)

    def __init__(self):
        self.state = self.HOVER
        self.start_action_time = get_time_ms()
        self.action_interval = 1200

    def update(self):
        if self.state == self.HOVER:
            if self.is_complete():
                self.state = self.LEFT_DOWN_RIGHT_UP

        elif self.state == self.LEFT_DOWN_RIGHT_UP:
            if self.is_complete():
                self.state = self.HOVER

    def is_complete(self):
        current_time = get_time_ms()
        if current_time - self.start_action_time >= self.action_interval:
            self.start_action_time = current_time
            return True
        return False


if __name__ == '__main__':
    api = UpAPI()
    controller = Controller()

    while True:

        controller.update()

        if controller.state == controller.HOVER:
            left_arm = data_arm.left_arm_clamp()
            right_arm = data_arm.right_arm_prepare_beat()
            api.execute_arm_action(left_arm, right_arm)

        elif controller.state == controller.LEFT_DOWN_RIGHT_UP:
            left_arm = data_arm.left_arm_clamp()
            right_arm = data_arm.right_arm_beat()
            api.execute_arm_action(left_arm, right_arm)
