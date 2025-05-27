from sdk.api import UpAPI
from sdk.logic_layer.spanner import Spanner
from sdk.logic_layer.time_meter import TimeMeter


if __name__ == '__main__':
    turn_rate = 100

    api = UpAPI()

    spanner = Spanner()
    span_waiter = TimeMeter(3500)

    spanner.start()
    span_waiter.start()
    print("开始旋转")

    while True:
        if span_waiter.complete():
            api.stop()
        else:
            api.spin_left(turn_rate)

   # left  ---- turn_rate = 50    time = 3700
   # right ---- turn_rate = -50   time = 3375
   # back  ---- turn_rate = 50    time = 7000

   # left  ---- turn_rate = 100   time = 1750
   # right ---- turn_rate = -100  time = 1750
   # back  ---- turn_rate = 100   time = 3500
