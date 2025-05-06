from ...utils import convert_util as tools


class ArmPositions:
    """
    单臂舵机数据
    """
    def __init__(self, hand, joint, arm, shoulder):
        self.hand = hand
        self.joint = joint
        self.arm = arm
        self.shoulder = shoulder


class UpperBody:
    """
    上半身舵机数据
    """

    def __init__(self, left_arm_positions, right_arm_positions, servo_move_time):
        # 左臂舵机数据
        self.left_hand = left_arm_positions.hand
        self.left_joint = left_arm_positions.joint
        self.left_arm = left_arm_positions.arm
        self.left_shoulder = left_arm_positions.shoulder

        # 腰部舵机数据
        self.waist_vertical = 0
        self.waist_horizontal = 0

        # 右臂舵机数据
        self.right_hand = right_arm_positions.hand
        self.right_joint = right_arm_positions.joint
        self.right_arm = right_arm_positions.arm
        self.right_shoulder = right_arm_positions.shoulder

        # 舵机移动时间
        self.servo_move_time = servo_move_time

    def to_list(self):
        left_hand_list = tools.decimal_convert_to_little_endian_list(self.left_hand)
        left_joint_list = tools.decimal_convert_to_little_endian_list(self.left_joint)
        left_arm_list = tools.decimal_convert_to_little_endian_list(self.left_arm)
        left_shoulder_list = tools.decimal_convert_to_little_endian_list(self.left_shoulder)

        waist_vertical_list = tools.decimal_convert_to_little_endian_list(self.waist_vertical)
        waist_horizontal_list = tools.decimal_convert_to_little_endian_list(self.waist_horizontal)

        right_hand_list = tools.decimal_convert_to_little_endian_list(self.right_hand)
        right_joint_list = tools.decimal_convert_to_little_endian_list(self.right_joint)
        right_arm_list = tools.decimal_convert_to_little_endian_list(self.right_arm)
        right_shoulder_list = tools.decimal_convert_to_little_endian_list(self.right_shoulder)

        left_position_list = left_hand_list + left_joint_list + left_arm_list + left_shoulder_list
        waist_position_list = waist_vertical_list + waist_horizontal_list
        right_position_list = right_hand_list + right_joint_list + right_arm_list + right_shoulder_list

        servo_move_time_list = tools.decimal_convert_to_little_endian_list(self.servo_move_time)

        position_list = left_position_list + waist_position_list + right_position_list + servo_move_time_list
        return position_list
