from sdk.api import UpAPI
import time

if __name__ == '__main__':
    api = UpAPI()

    while True:
        data = api.get_grayscale_data()
        print(data)

        time.sleep(0.5)
