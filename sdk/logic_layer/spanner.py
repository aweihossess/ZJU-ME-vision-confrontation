from enum import Enum, auto


class State(Enum):
    SET_OUT = auto()
    TURN = auto()
    STOP = auto()


class Spanner:
    def __init__(self):
        self.state = State.SET_OUT
        self.target_count = 1  # 目标旋转次数（在十字上旋转90°的次数）
        self.reach_count = 0
        self.alignment_number = 6

    def __update_state(self, data):
        if self.state == State.SET_OUT:
            if not data[0] and not data[1] and not data[5] and not data[6]:
                self.state = State.TURN

        elif self.state == State.TURN:
            if self.__has_consecutive_trues(data):
                self.reach_count += 1
                if self.reach_count >= self.target_count:
                    self.state = State.STOP
                else:
                    self.state = State.SET_OUT

    def __has_consecutive_trues(self, data):
        for i in range(len(data) - self.alignment_number + 1):
            if all(data[i:i + self.alignment_number]):
                return True
        return False

    def start(self, target_count=1):
        """
        开始旋转
        :param target_count: 在十字上旋转90°的次数
        """
        self.state = State.SET_OUT
        self.target_count = target_count
        self.reach_count = 0

    def complete(self, data):
        if len(data) != 7:
            raise RuntimeError("grayscale data length is not 7")

        self.__update_state(data)
        return self.state == State.STOP