from sdk.api import UpAPI
import time


def get_time_ms():
    return int(time.time() * 1000)


class Controller:
    IDLE, FORWARD, BACK, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, FINISH = range(8)

    def __init__(self):
        self.state = self.IDLE
        self.start_action_time = get_time_ms()
        self.action_interval = 1000

    def update(self):
        if self.state == self.IDLE:
            if self.is_complete():
                self.state = self.FORWARD

        elif self.state == self.FORWARD:
            if self.is_complete():
                self.state = self.BACK

        elif self.state == self.BACK:
            if self.is_complete():
                self.state = self.MOVE_LEFT

        elif self.state == self.MOVE_LEFT:
            if self.is_complete():
                self.state = self.MOVE_RIGHT

        elif self.state == self.MOVE_RIGHT:
            if self.is_complete():
                self.state = self.TURN_LEFT

        elif self.state == self.TURN_LEFT:
            if self.is_complete():
                self.state = self.TURN_RIGHT

        elif self.state == self.TURN_RIGHT:
            if self.is_complete():
                self.state = self.FINISH

        elif self.state == self.FINISH:
            pass

    def is_complete(self):
        current_time = get_time_ms()
        if current_time - self.start_action_time >= self.action_interval:
            self.start_action_time = current_time
            return True
        return False


if __name__ == '__main__':
    move_speed = 10
    spin_speed = 50

    api = UpAPI()
    controller = Controller()

    while True:

        controller.update()

        if controller.state == Controller.IDLE:
            pass

        elif controller.state == Controller.FORWARD:
            api.move_forward(move_speed)

        elif controller.state == Controller.BACK:
            api.move_backward(move_speed)

        elif controller.state == Controller.MOVE_LEFT:
            api.move_left(move_speed)

        elif controller.state == Controller.MOVE_RIGHT:
            api.move_right(move_speed)

        elif controller.state == Controller.TURN_LEFT:
            api.spin_left(spin_speed)

        elif controller.state == Controller.TURN_RIGHT:
            api.spin_right(spin_speed)

        elif controller.state == Controller.FINISH:
            print("finish")
            break
