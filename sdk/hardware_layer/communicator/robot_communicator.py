class RobotCommunicator:

    def __init__(self, serial_port):
        """ 初始化 """
        self.serial_port = serial_port
        self.read_buffer = bytearray()
        self.frame_header = [0xF5, 0x5F]

    def send_command(self, byte_array):
        # print(f"robot send: {byte_array}")
        # 发送帧
        self.serial_port.write(byte_array)

    def receive_packet(self):

        # 如果串口中有数据，读取到缓冲区
        if self.serial_port.in_waiting > 0:
            serial_data = self.serial_port.read(self.serial_port.in_waiting)
            self.read_buffer.extend(serial_data)

        # 检查缓冲区中是否有足够的数据
        # 长度：帧头[2字节]  id[1字节]  数据包长度[1字节]  响应指令[1字节]  请求指令[1字节]  数据[至少1字节]  校验[1字节]
        while len(self.read_buffer) >= 8:
            # 找到帧头
            if self.read_buffer[0:2] == self.frame_header:
                # 数据长度（第4个字节）
                packet_length = self.read_buffer[4]
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

    def read_response(self):
        response = bytearray()

        while self.serial_port.in_waiting > 0:
            response += self.serial_port.read(self.serial_port.in_waiting)

        if response:
            print("接收到的响应:", ' '.join(format(x, '02X') for x in response))
        else:
            print("未接收到响应。")

        return response

    def __calculate_checksum(self, data):
        return ~sum(data) & 0xFF
