import time
from robotpi_Cmd import UPComBotCommand
from robotpi_serOp import serOp


class Movement:

    isOpen = True

    def __init__(self):
        self.isOpen = True
        self.cmd = UPComBotCommand()
        self.action = serOp()
        '''
        # _______________move_______________
        # (direction, speed, turn_speed, time)
        # move_forward(0, 10, 0, 500)
        # move_left(90, 10, 0, 500)
        # move_right(270, 10, 0, 500)
        # move_backward(180, 10, 0, 500)
        # turn_left(0, 0, 100, 500)
        # turn_right(0, 0, -100, 500)
        # left_ward(0, 5, 50, 500)         # going left while moving forward
        # right_ward(0, 5, -50, 500)       # going right while moving forward
        # _______________hands______________
        # hands position(device:1, 2, 3, 4, 31,32,33,34; position:0-4096; speed:0-10)
        # left_hand(4 1100 10)(34 1300 10)  # left hand up, right hand down.
        # right_hand(4 2700 10)(34 3000 10) # right hand up, left hand down.
        # rise_hands(4 1100 10)(34 3000 10) # both hands up.
        # _______________voice_____________
        # set_volume(vol)           # vol range:'0-31'  
        # play_sound(folder, track) # self defined folder name is 255
          file name:'1:刘备'、'2:关羽'、'3:张飞'、'10:左转'、'11:右转'、'12:前进'、'13:停止'
        '''

    def move_forward(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(0, speed, 0, times)
            self.action.write_serial(command)
            return True
        return False

    def move_left(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(90, speed, 0, times)
            self.action.write_serial(command)
            return True
        return False

    def move_right(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(270, speed, 0, times)
            self.action.write_serial(command)
            return True
        return False

    def move_backward(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(180, speed, 0, times)
            self.action.write_serial(command)
            return True
        return False

    def turn_left(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(0, 0, speed*10, times)
            self.action.write_serial(command)
            return True
        return False

    def turn_right(self, speed=10, times=500):
        if self.isOpen:
            command = self.cmd.Command(0, 0, -speed*10, times)
            self.action.write_serial(command)
            return True
        return False

    def stop(self):
        if self.isOpen:
            command = self.cmd.Command(0, 0, 0, 500)
            self.action.write_serial(command)
            return True
        return False

    def seeking(self, speed=10, times=3000):
        if self.isOpen:

            command = self.cmd.Command(0, 0, -speed, times)

            self.action.write_serial(command)

    def wave_hands(self): 
        if self.isOpen:
            command = self.cmd.wave_hands()
            self.action.write_serial(command)
            time.sleep(1)
            return True
        return False

    def hit(self):
        if self.isOpen:
            command = self.cmd.hit()
            self.action.write_serial(command)
            time.sleep(3)
            return True
        return False

    def left_ward(self):
        if self.isOpen:
            command = self.cmd.Command(0, 5, 50, 500)
            self.action.write_serial(command)
            return True
        return False

    def right_ward(self):
        if self.isOpen:
            command = self.cmd.Command(0, 5, -50, 500)
            self.action.write_serial(command)
            return True
        return False

    def left_hand(self):
        if self.isOpen:
            device = 4 
            position = 1100 # 0-4096
            speed_L = 10 
            speed_H = 0 
            

            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            device = 34
            position = 1300  # 0-4096
            speed_L = 10
            speed_H = 0

            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            return True
        return False

    def right_hand(self):
        if self.isOpen:
            device = 4 
            position = 2700 # 0-4096
            speed_L = 10 
            speed_H = 0 
            

            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF7

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            device = 34
            position = 3000  # 0-4096
            speed_L = 10
            speed_H = 0

            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            return True
        return False
    
    def rise_hands(self):
        if self.isOpen:
            device = 4 
            position = 1100 # 0-4096
            speed_L = 10 
            speed_H = 0 
            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            device = 34
            position = 3000  # 0-4096
            speed_L = 10
            speed_H = 0

            data = [0] * 5
            data[0] = device & 0xFF
            data[2] = position & 0xFF
            data[1] = (position >> 8) & 0xFF
            data[4] = speed_L & 0xFF
            data[3] = speed_H & 0xFF

            buffer, len = self.cmd.GenerateCmd(0x07, 0x5C, 0x05, data)
            self.action.write_serial(buffer)

            return True
        return False

    def set_volume(self, vol):
        data = [0] * 1
        data[0] = vol & 0xFF

        buffer, len = self.cmd.GenerateCmd(0x01, 0x08, 0x01, data)
        self.action.write_serial(buffer)

    def play_sound(self, folder, track):
        data = [0] * 2
        data[0] = folder & 0xFF
        data[1] = track & 0xFF

        buffer, len = self.cmd.GenerateCmd(0x01, 0x42, 0x02, data)
        self.action.write_serial(buffer)

    def hold(self):
        name = 'hold'
        buffer = self.cmd.call_action_by_name(name)
        self.action.write_serial(buffer)

    def play_sounds(self, filename):
        b_name = filename.encode("GBK")
        l = len(b_name) & 0xFF
        data = [0] * l
        for i in range(l):
            data[i] = b_name[i]
        buffer, _ = self.cmd.GenerateCmd(0x01, 0x70, l, data)
        self.action.write_serial(buffer)

    # def hit(self):
    #     name = 'hit'
    #     buffer = self.cmd.call_action_by_name(name)
    #     self.action.write_serial(buffer)


if __name__ == '__main__':
    mv = Movement()
    # mv.wave_hands()
    mv.hit()
    mv.set_volume(20)
    mv.play_sounds('0255\\0001.WAV')
    while True:
        print("Testing...")
        mv.hold()
        break
        # recv_data = mv.action.serial_listen()
        # if recv_data:
        #     #mv.wave_hands()
        #     for i in recv_data:
        #         print("data received:", hex(i))
        #     print("____")
        #     break
        #
        #

