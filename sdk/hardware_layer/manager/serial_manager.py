import serial, threading


class SerialManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # 第一次检查避免不必要的锁开销
        if cls._instance is None:
            with cls._lock:
                # 第二次检查确保只有一个实例被创建
                if cls._instance is None:
                    instance = super(SerialManager, cls).__new__(cls)
                    instance.__initialize_serial_ports()
                    cls._instance = instance
        return cls._instance

    def __initialize_serial_ports(self):
        # 初始化串口连接
        self.__serial_uart = serial.Serial(
            port="/dev/ttyS0",
            baudrate=115200,
            bytesize=8,
            parity='E',
            stopbits=1,
            timeout=2
        )

        self.__serial_usb = serial.Serial(
            port="/dev/ttyUSB0",
            baudrate=500000,
            bytesize=8,
            stopbits=1,
            timeout=2
        )

        # 检查端口状态
        if not self.__serial_uart.is_open:
            raise IOError('/dev/ttyS0 is not open')
        if not self.__serial_usb.is_open:
            raise IOError('/dev/ttyUSB0 is not open')

    def __getattr__(self, name):
        """ 委托属性访问到实际实例 """
        return getattr(self._instance, name)

    def get_serial_uart(self):
        return self.__serial_uart

    def get_serial_usb(self):
        return self.__serial_usb
