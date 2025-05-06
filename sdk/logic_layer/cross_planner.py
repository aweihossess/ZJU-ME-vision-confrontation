class CrossLocator:
    """
    十字定位器
    """

    def __init__(self):
        self.target_data = [True, True, True, True, True, True, True]

    def leave_cross(self, data):
        if len(data) == 7:
            return (
                    data[1] != self.target_data[1]
                    and data[2] != self.target_data[2]
                    and data[3] != self.target_data[3]
                    and data[4] != self.target_data[4]
                    and data[5] != self.target_data[5]
                    and data[0] != self.target_data[0]
                    and data[6] != self.target_data[6]
            )
        return False

    def translate_to_center(self, data):
        if len(data) == 7:
            center_data = data[3] == self.target_data[3]
            return center_data
        return False

    def reach_target(self, data, high_precision=True):
        if len(data) == 7:
            edge = data[0] == self.target_data[0] or data[6] == self.target_data[6]

            center = (data[1] == self.target_data[1]
                    and data[2] == self.target_data[2]
                    and data[3] == self.target_data[3]
                    and data[4] == self.target_data[4]
                    and data[5] == self.target_data[5])

            if high_precision:
                return center and edge
            else:
                return center
        return False

    def move_straight(self, data):
        if len(data) == 7:
            return (data[0] != self.target_data[0]
                    and data[1] != self.target_data[1]
                    and data[2] != self.target_data[2]
                    and data[3] != self.target_data[3]
                    and data[4] != self.target_data[4]
                    and data[5] != self.target_data[5]
                    and data[6] != self.target_data[6]
                    )
        return False

    def detect_black(self, data):
        if len(data) == 7:
            return (
                    data[0] == self.target_data[0]
                    or data[1] == self.target_data[1]
                    or data[2] == self.target_data[2]
                    or data[3] == self.target_data[3]
                    or data[4] == self.target_data[4]
                    or data[5] == self.target_data[5]
                    or data[6] == self.target_data[6]
            )
        return False

    def seeking_left(self, data):
        if len(data) == 7:
            grayscale0 = data[0] == self.target_data[0]
            grayscale1 = data[1] == self.target_data[1]
            grayscale5 = data[5] == self.target_data[5]
            grayscale6 = data[6] == self.target_data[6]

            return (((not grayscale5 and not grayscale6) and grayscale0) or
                    ((not grayscale5 and not grayscale6) and grayscale1))
        return False

    def seeking_right(self, data):
        if len(data) == 7:
            grayscale0 = data[0] == self.target_data[0]
            grayscale1 = data[1] == self.target_data[1]
            grayscale5 = data[5] == self.target_data[5]
            grayscale6 = data[6] == self.target_data[6]

            return (((not grayscale0 and not grayscale1) and grayscale5) or
                    ((not grayscale0 and not grayscale1) and grayscale6))
        return False

    def translate_left(self, data):
        if len(data) == 7:
            grayscale0 = data[0] == self.target_data[0]
            grayscale1 = data[1] == self.target_data[1]
            grayscale2 = data[2] == self.target_data[2]
            grayscale3 = data[3] == self.target_data[3]

            return grayscale0 and grayscale1 and grayscale2 and grayscale3
        return False

    def translate_right(self, data):
        if len(data) == 7:
            grayscale3 = data[3] == self.target_data[3]
            grayscale4 = data[4] == self.target_data[4]
            grayscale5 = data[5] == self.target_data[5]
            grayscale6 = data[6] == self.target_data[6]

            return grayscale3 and grayscale4 and grayscale5 and grayscale6
        return False

    def move_left(self, data):
        if len(data) == 7:
            return any(data[:3])
        return False

    def move_right(self, data):
        if len(data) == 7:
            return any(data[4:])
        return False

    def turn_left(self, data):
        if len(data) == 7:
            pass
        return False
