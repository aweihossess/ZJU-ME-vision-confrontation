from sdk.api import UpAPI
from sdk.logic_layer.pid import PIDController


if __name__ == '__main__':
    # 参数设置
    move_speed = 16

    k_p = 16
    k_i = 0.01
    k_d = 140

    # 传感器 逻辑处理器 执行器
    api = UpAPI()
    pid = PIDController(k_p=k_p, k_i=k_i, k_d=k_d)

    while True:
        # 获取数据
        grayscale_data = api.get_grayscale_data()

        # 计算偏移量
        offset = api.follow_line()

        # pid
        turn_rate = pid.compute(offset)

        # print("=============")
        # print(f"grayscale_data: {grayscale_data}")
        # print(f"offset: {offset}")
        # print(f"turn_rate: {turn_rate}")

        # 发送移动指令
        if offset is not None:
            api.move_rotation(speed=move_speed, turn_rate=turn_rate)
        else:
            api.move_forward(speed=move_speed)
