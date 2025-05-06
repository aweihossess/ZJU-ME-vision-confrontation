from sdk.api import UpAPI


if __name__ == '__main__':
    api = UpAPI()

    raise_left_arm_id = 1
    raise_right_arm_id = 3
    raise_arms_id = 5

    while True:

        find_apriltag, tag_id, offset_x = api.detect_apriltag()

        if find_apriltag:
            print(f"tag_id: {tag_id}")

            if tag_id == raise_left_arm_id:
                api.raise_left_arm()

            elif tag_id == raise_right_arm_id:
                api.raise_right_arm()

            elif tag_id == raise_arms_id:
                api.raise_arms()

            else:
                api.put_down_arms()

        else:
            api.put_down_arms()