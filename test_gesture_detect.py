from sdk.api import UpAPI


if __name__ == '__main__':
    raise_arms_number = 5

    api = UpAPI()

    while True:
        # 获取数据
        find_gesture, number = api.detect_gesture()

        if find_gesture:
            print(f"number: {number}")

            if number == raise_arms_number:
                api.raise_arms()

            else:
                api.put_down_arms()

        else:
            api.put_down_arms()
