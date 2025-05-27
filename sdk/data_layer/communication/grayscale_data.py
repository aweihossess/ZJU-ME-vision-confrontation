class __GrayscaleCommand:
    def __init__(self, command, address, parameters):
        self.head1 = 0xFE
        self.head2 = 0xEF
        self.id = 0x06
        self.command = command
        self.address = address
        self.parameters = parameters
        self.data_length = 3 + len(parameters)
        self.checksum = self.__calculate_checksum()

    def __calculate_checksum(self):
        sum_data = self.id + self.command + self.address + sum(self.parameters) + self.data_length
        return ~sum_data & 0xFF

    def to_bytes(self):
        frame = bytearray()
        frame.append(self.head1)
        frame.append(self.head2)
        frame.append(self.id)
        frame.append(self.data_length)
        frame.append(self.command)
        frame.append(self.address)
        for parameter in self.parameters:
            frame.append(parameter)
        frame.append(self.checksum)
        return frame


class GrayscaleReadCommand(__GrayscaleCommand):
    def __init__(self, address, parameters=[]):
        super().__init__(0x02, address, parameters)

    def __str__(self):
        return f"GrayscaleReadCommand(head1={self.head1}, head2={self.head2}, id={self.id}, length={self.data_length}, command={self.command}), address={self.address}, parameters={self.parameters}, checksum={self.checksum})"


class GrayscaleWriteCommand(__GrayscaleCommand):
    def __init__(self, address, parameters=[]):
        super().__init__(0x03, address, parameters)

    def __str__(self):
        return f"GrayscaleWriteCommand(head1={self.head1}, head2={self.head2}, id={self.id}, length={self.data_length}, command={self.command}), address={self.address}, parameters={self.parameters}, checksum={self.checksum})"
