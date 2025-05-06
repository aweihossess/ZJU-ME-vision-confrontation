class __RobotCommand:
    types = {
        "servo": 0x07,
        "chassis": 0x08
    }

    def __init__(self, device_type, command, parameters):
        self.head1 = 0xF5
        self.head2 = 0x5F
        self.device_type = device_type
        self.command = command
        self.data_length = len(parameters)
        self.parameters = parameters
        self.checksum = self.__calculate_checksum()

    def __calculate_checksum(self):
        sum_data = self.device_type + self.command + self.data_length + sum(self.parameters)
        return ~sum_data & 0xFF

    def to_bytes(self):
        frame = bytearray()
        frame.append(self.head1)
        frame.append(self.head2)
        frame.append(self.device_type)
        frame.append(self.command)
        frame.append(self.data_length)
        for parameter in self.parameters:
            frame.append(parameter)
        frame.append(self.checksum)
        return frame


class RobotServoCommand(__RobotCommand):
    def __init__(self, command, parameters=[]):
        super().__init__(self.types["servo"], command, parameters)

    def __str__(self):
        return f"RobotServoCommand(head1={self.head1}, head2={self.head2}, command={self.command}, length={self.data_length}, parameters={self.parameters}, checksum={self.checksum})"


class RobotChassisCommand(__RobotCommand):
    def __init__(self, command, parameters=[]):
        super().__init__(self.types["chassis"], command, parameters)

    def __str__(self):
        return f"RobotChassisCommand(head1={self.head1}, head2={self.head2}, device={self.device_type}, command={self.command}, length={self.data_length}, parameters={self.parameters}, checksum={self.checksum})"
