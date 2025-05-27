import time


class TimeMeter:
    def __init__(self, move_interval):
        self.move_interval = move_interval
        self.last_move_time = self.__get_time_ms()
        self.in_progress = False

    def start(self):
        self.last_move_time = self.__get_time_ms()
        self.in_progress = True

    def complete(self):
        if self.__get_time_ms() - self.last_move_time >= self.move_interval:
            self.in_progress = False
            return True
        return False

    def __get_time_ms(self):
        return int(time.time() * 1000)