from sdk.data_layer.arm import arm_action_factory as arm_data
from sdk.api import UpAPI


if __name__ == '__main__':
    api = UpAPI()
    arm_reset = {
        "left": arm_data.left_arm_clamp(),
        "right": arm_data.right_arm_clamp()
    }
    api.execute_arm_action(arm_reset["left"], arm_reset["right"])
