from ..utils import convert_util as tools
from ..hardware_layer.communicator.grayscale_communicator import GrayscaleCommunicator
from .communication.grayscale_data import GrayscaleReadCommand


class Grayscale:
    __addresses = {
        'get_grayscale_data': 0x14,
    }

    def __init__(self, serial_port, threshold=3000):
        self.communicator = GrayscaleCommunicator(serial_port)
        self.threshold = threshold

    def get_grayscale_data(self):
        command = GrayscaleReadCommand(address=self.__addresses['get_grayscale_data'], parameters=[0x0E])
        data = self.communicator.get_response_data(command.to_bytes())
        if data:
            data_list = self.__separate_data(data)
            data_list = [tools.little_endian_list_convert_to_decimal(data) for data in data_list]
            # print(f"data_list: {data_list}")
            # data_list = self.__binary(data_list)
            return data_list
        else:
            return None

    def __separate_data(self, data, byte_count=2):
        """
        将 bytearray 数据分离成 10 进制数组
        :param data: 待分离 bytearray 数据
        :param byte_count: bytearray 数据分割步长
        :return: 10 进制数组
        """
        data_length = len(data)
        if data_length % byte_count != 0:
            raise ValueError("Divisible Error \n data_length: {data_length}, byte_count: {byte_count]}")

        separated_data = [data[i:i + 2] for i in range(0, data_length, byte_count)]
        return separated_data
