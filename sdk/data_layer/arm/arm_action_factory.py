from .arm_data import ArmPositions


def left_arm_raise():
    return ArmPositions(hand=2150, joint=2150, arm=650, shoulder=1200)


def right_arm_raise():
    return ArmPositions(hand=2150, joint=2200, arm=3600, shoulder=2900)


def left_arm_down():
    return ArmPositions(hand=2150, joint=2150, arm=650, shoulder=2670)


def right_arm_down():
    return ArmPositions(hand=2150, joint=2200, arm=3600, shoulder=1400)


def left_arm_open():
    return ArmPositions(hand=2140, joint=3440, arm=1900, shoulder=2200)


def right_arm_open():
    return ArmPositions(hand=3400, joint=600, arm=2030, shoulder=2200)


def left_arm_hug():
    return ArmPositions(hand=0, joint=0, arm=500, shoulder=1900)


def right_arm_hug():
    return ArmPositions(hand=0, joint=0, arm=1900, shoulder=2150)


def left_arm_hover():
    return ArmPositions(hand=2150, joint=2200, arm=600, shoulder=2200)


def right_arm_hover():
    return ArmPositions(hand=2150, joint=2150, arm=3300, shoulder=2200)


def right_arm_prepare_beat():
    return ArmPositions(hand=1200, joint=1800, arm=3500, shoulder=2500)


def right_arm_beat():
    return ArmPositions(hand=3000, joint=1800, arm=3500, shoulder=2050)


def left_arm_clamp():
    return ArmPositions(hand=2048, joint=2048, arm=620, shoulder=2670)


def right_arm_clamp():
    return ArmPositions(hand=600, joint=2048, arm=3500, shoulder=1400)