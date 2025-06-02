import time
from sdk.logic_layer.pid import PIDController
from sdk.logic_layer.cross_planner import CrossLocator

def follow_line_to_home(robot, target_cross_count):
    """
    沿线行驶直到回家（经过指定数量的十字）
    
    :param robot: RobotBody实例
    :param target_cross_count: 需要经过的十字数量
    :return: 无
    """
    # 初始化PID控制器和十字定位器
    pid = PIDController(k_p=15, k_i=0.2, k_d=1.0)
    locator = CrossLocator()
    
    # 巡线参数
    speed_follow_line = 10  # 巡线前进移动速度
    speed_locate_move = 4   # 定位移动速度
    speed_locate_turn = 20  # 定位旋转速度
    
    cross_count = 0
    count_out_line = 0
    count_max_out_line = 5
    
    print("开始巡线回家...")
    
    while True:
        # 获取灰度数据
        grayscale_data = robot.api.get_grayscale_data()
        
        # 检测是否在线上
        if locator.detect_black(grayscale_data):
            # 在黑线上
            if locator.translate_to_center(grayscale_data):
                # 中心对齐了
                if locator.reach_target(grayscale_data, False):
                    # 到达十字中心
                    print(f"检测到十字，当前第 {cross_count+1} 个十字")
                    cross_count += 1
                    # 停下来调整位置
                    robot.api.stop()
                    # 短暂等待
                    time.sleep(0.5)
                    
                    if cross_count >= target_cross_count:
                        print(f"到达目标十字数量 {target_cross_count}，回家完成")
                        robot.api.stop()
                        return
                    else:
                        # 继续前进
                        print("继续前进寻找下一个十字")
                        robot.api.move_forward(speed_follow_line)
                else:
                    # 中心对齐但未到达十字中心，继续前进或调整方向
                    if locator.seeking_left(grayscale_data):
                        print("中心是黑色了，前进左转")
                        robot.api.spin_left(speed_locate_turn)
                    elif locator.seeking_right(grayscale_data):
                        print("中心是黑色了，前进右转")
                        robot.api.spin_right(speed_locate_turn)
                    else:
                        print("中心是黑色，继续前进")
                        robot.api.move_forward(speed_locate_move)
            else:
                # 中心未对齐，需要左右平移调整
                if locator.move_straight(grayscale_data):
                    print("All False")
                    robot.api.move_forward(speed_locate_move)
                else:
                    if locator.move_left(grayscale_data):
                        print("中心未检测到黑色，左平移")
                        robot.api.move_left(speed_locate_move)
                    elif locator.move_right(grayscale_data):
                        print("中心未检测到黑色，右平移")
                        robot.api.move_right(speed_locate_move)
                    else:
                        print("无法判断方向")
            
            # 重置脱线计数
            count_out_line = 0
        else:
            # 脱离黑线处理
            count_out_line += 1
            if count_out_line > count_max_out_line:
                # 尝试找回黑线
                print("脱离黑线，尝试找回...")
                # 获取线偏移
                line_center_offset = robot.api.follow_line()
                if line_center_offset > 0:
                    # 偏右，向左转
                    robot.api.spin_left(30)
                else:
                    # 偏左，向右转
                    robot.api.spin_right(30)
                time.sleep(0.1)
                count_out_line = 0
            else:
                # 使用PID控制巡线
                line_center_offset = robot.api.follow_line()
                turn_rate = pid.compute(line_center_offset)
                robot.api.move_rotation(speed=speed_follow_line, turn_rate=turn_rate)
                print(f"巡线中，偏移量: {line_center_offset}, 转向率: {turn_rate}") 