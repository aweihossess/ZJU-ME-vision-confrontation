import serial

class serOp():
    ser = serial.Serial(
            port="/dev/ttyUSB0",
            baudrate=115200,
            bytesize=8,
            parity='E',
            stopbits=1,
            timeout=2)
    isOpen = True

    def __int__(self):
        self.isOpen = True
        
    def open(self):
        self.ser.open()
        if(serOp.ser.isOpen):
            self.isOpen = True
            print ("open")
        else:
            self.isOpen = False

    def serial_listen(self):
        data = []
        while serOp.ser.inWaiting() > 0:
            k = serOp.ser.read()
            data.append(int.from_bytes(k, byteorder='little', signed=False))
        return data

    def serial_listen_check(self):
        len_limit = 0
        while serOp.ser.inWaiting() > 0:
            h = serOp.ser.read()
            hey = int.from_bytes(h, byteorder='big', signed=False)
            if hey == 245:
                data = [hey]
                while serOp.ser.inWaiting() > 0:
                    k = serOp.ser.read()
                    data.append(int.from_bytes(k, byteorder='big', signed=False))
                    len_limit += 1
                    if len(data) >= 5:
                        check = data[2]
                        check = check+data[3]
                        check = check+data[4]
                        for i in range(data[4]):
                            s = serOp.ser.read()
                            if s is None:
                                print("broken package.")
                                break
                            e = int.from_bytes(s, byteorder='big', signed=False)
                            data.append(e)
                            check = check + e
                        ending = serOp.ser.read()
                        end = int.from_bytes(ending, byteorder='big', signed=False)
                        data.append(end)
                        dt = data[5:-1]
                        charactor = []
                        for ss in dt:
                            charactor.append(chr(ss))
                        print(charactor)
                        if data[-1] == (~check) & 0xFF:
                            print("test ok")
                            return data
                        else:
                            break
                len_limit = 0
                print("not OK, try again.")


    def serial_listening(self):

        data = ''
        data = data.encode()
        while self.isOpen:
            n = serOp.ser.inWaiting()
            if n:
                data = data + self.ser.read(n)
            if len(data) > 0 and n == 0:
                return data

    def serial_string(self):
        data = b''
        while serOp.ser.inWaiting() > 0:
            k = serOp.ser.read()
            data += k
        return data[5:-1]

    def write_serial(self, command):
        self.ser.write(command)


if __name__ == '__main__':
    from robotpi_Cmd import UPComBotCommand
    import time
    com = UPComBotCommand()
    ser = serOp()
    while True:
        # name = '01.00.Welcome_guy'
        # print("name:", name)
        # b_name = name.encode()
        # command, _ = com.GenerateCmd(device=0x09, cmd=0x4B, len=len(b_name), data=b_name)
        # print("original data:",  command)
        # ser.write_serial(command)

        # test = [0] * 1
        # test[0] = 2 & 0xFF
        # send_data, _ = com.GenerateCmd(device=0x09, cmd=0x4B, len=0x00, data=None)
        # print("origin data:", send_data)
        # ser.write_serial(send_data)

        recv_data = ser.serial_listen()

        if recv_data:
            #mv.wave_hands()
            for i in recv_data:
                print("data received:", hex(i))
            print("____")
            dt = recv_data[5:-1]
            charactor = []
            for ss in dt:
                charactor.append(chr(ss))
            print(charactor)



        # recv_data = ser.serial_listen_check()
        #
        # if recv_data:
        #     #mv.wave_hands()
        #     for i in recv_data:
        #         print("data received:", hex(i))
        #     print("____")

