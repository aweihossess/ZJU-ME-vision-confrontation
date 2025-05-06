class SingleLineFollower:
    def __init__(self, mid_index=3, data_length=7):
        self.mid_index = mid_index
        self.data_length = data_length

    def process_frame(self, data):
        """
        通过滑动窗口法，处理数据帧
        :param: data: 灰度阵列数字量数据
        :return: 旋转偏差，左正右负
        """
        if len(data) != self.data_length:
            raise Exception(f"single line follower data length is not {self.data_length}.")

        longest_segment = []
        current_segment = []

        for index, value in enumerate(data):
            if value:
                current_segment.append(index)
            else:
                if len(current_segment) > len(longest_segment):
                    longest_segment = current_segment
                current_segment = []

        # 检查最后一个段
        if len(current_segment) > len(longest_segment):
            longest_segment = current_segment

        if longest_segment:
            avg_indices = self.mid_index - (sum(longest_segment) / len(longest_segment))
        else:
            avg_indices = 0

        return avg_indices
