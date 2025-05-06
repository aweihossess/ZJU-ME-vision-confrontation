import time


class GrayscaleCommunicator:

    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.read_buffer = bytearray()
        self.frame_header = [0xFE, 0xEF]

    def send_command(self, byte_array):
        # print(f"chassis send: {byte_array}")
        # 发送帧
        self.serial_port.write(byte_array)

    def receive_packet(self):
        # 如果串口中有数据，读取到缓冲区
        if self.serial_port.in_waiting > 0:
            serial_data = self.serial_port.read(self.serial_port.in_waiting)
            self.read_buffer.extend(serial_data)

        # 检查缓冲区中是否有足够的数据
        while len(self.read_buffer) >= 8:
            # 找到帧头
            if self.read_buffer[0] == self.frame_header[0] and self.read_buffer[1] == self.frame_header[1]:
                # 数据长度（第4个字节）
                packet_length = self.read_buffer[3]
                total_length = 4 + packet_length  # 全部长度（从帧头一直到校验位）

                # 检查缓冲区是否包含完整数据
                if len(self.read_buffer) >= total_length:
                    packet = self.read_buffer[:total_length]  # 提取完整数据包
                    self.read_buffer = self.read_buffer[total_length:]  # 移除已读取的数据包

                    # 校验数据包
                    checksum = self.__calculate_checksum(packet[2:-1])
                    if checksum == packet[-1]:
                        return packet
                    else:
                        print('Checksum mismatch, dropping packet.')
                        self.read_buffer.pop(0)  # 丢弃帧头，继续寻找下一个包
            else:
                # 非帧头，丢弃第一个字节
                self.read_buffer.pop(0)

        # 如果没有完整数据包，返回 None
        return None

    def get_response_data(self, data):
        self.send_command(data)
        time.sleep(0.01)
        packet = self.receive_packet()
        if packet is not None:
            if self.__parse_response(packet):
                data = packet[6:-1]
                return data
        return None

    def __calculate_checksum(self, data):
        return ~sum(data) & 0xFF

    def __parse_response(self, response):
        header = response[0:2]
        id = response[2]
        length = response[3]
        command = response[4]
        address = response[5]
        data = response[6:-1]
        checksum = response[-1]

        # print(f"解析响应: ID={id:02x}, 指令={command:02x}, 地址={address:02x}, 数据={data.hex()}, 校验和={checksum}")

        response_checksum = self.__calculate_checksum(response[2:-1])
        result = response_checksum == checksum
        # print("返回校验: ", result)

        return result
