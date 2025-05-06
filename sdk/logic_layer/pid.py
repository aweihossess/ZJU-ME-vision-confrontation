class PIDController:
    def __init__(self, k_p=15.0, k_i=0.2, k_d=1.0, max_output=100, integral_limit=100):
        # 比例系数 (建议范围 10-30)
        self.k_p = k_p
        # 积分系数 (建议范围 0.1-0.5)
        self.k_i = k_i
        # 微分系数 (建议范围 0.5-2)
        self.k_d = k_d
        # 输出限幅 (±max_output)
        self.max_output = max_output
        # 积分限幅 (±integral_limit)
        self.integral_limit = integral_limit

        # 运行参数
        self.prev_error = 0.0
        self.integral = 0.0
        #下面是原来的参数
        # self.prev_error = 0.0
        # self.integral = 0.0
    def compute(self, offset):
        # 当前误差（输入已经是偏移量）
        error = offset

        # 比例项
        proportion = self.k_p * error

        # 积分项（带限幅）
        self.integral += error
        print(f"integral: {self.integral}")
        # 积分限幅
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        integration = self.k_i * self.integral

        # 微分项（基于误差变化率）
        differentiation = self.k_d * (error - self.prev_error)
        self.prev_error = error

        # 计算总输出
        output = proportion + integration + differentiation

        # 输出限幅
        if output > self.max_output:
            output = self.max_output
        elif output < -self.max_output:
            output = -self.max_output

        return int(round(output))