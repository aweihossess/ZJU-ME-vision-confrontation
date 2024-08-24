import os
import time
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
from robotPi import robotPi
from rev_cam import rev_cam

start = 4# 1 2 3 4 5 6

#time_left = 500
threshold = 135 # 二值化阈值, 越小越白
threshold2 = 240 # 靶子的二值化阈值 

black_pixels_micai = 1000
black_pixels_back = 1500
black_pixels_qipan = 1300
#终点
min_r = 5
max_r = 60
target_x = 280
target_r = 55 #停下半径大小
tolerant_x = 15 #圆心坐标允许误差

width1, height1 = 160, 120
width2, height2 = 480, 360

def auto_pilot():
    time2 = 60 # 进入迷彩区
    time3 = 60 # 开始找棋盘格
    cap1 = cv2.VideoCapture(0)  # 胸前的摄像头
    robot = robotPi()

    #定义模板
    template_imgs = []
    template_imgs.append(cv2.imread("template/forward.jpg",0))
    template_imgs.append(cv2.imread("template/left.jpg",0))
    template_imgs.append(cv2.imread("template/right.jpg",0))
    template_imgs.append(cv2.imread("template/left1.jpg",0))
    '''
    forward_imgs = []
    forward_imgs.append(cv2.imread("template/forward.jpg",0))
    forward_imgs.append(cv2.imread("template/sleft.jpg",0))
    forward_imgs.append(cv2.imread("template/sright.jpg",0))
    '''
    #起点转正
    if(start == 1):
        robot.movement.turn_right(times = 1900)
        time.sleep(1.9)
    elif(start == 2):
        robot.movement.turn_right(times = 1000)
        time.sleep(1)
    elif(start == 3):
        robot.movement.turn_right(times = 300)
        time.sleep(0.3)
    elif(start == 4):
        #robot.movement.turn_left(speed=6,times=400)
        robot.movement.turn_left(times = 700)
        time.sleep(0.7)
    elif(start == 5):
        robot.movement.turn_left(times = 1300)
        time.sleep(1.3)
    elif(start == 6):
        robot.movement.turn_left(times = 2300)
        time.sleep(2.3)

    start_time = time.time()
    isback = 0
    micai = 0
    while cap1.isOpened():
        #图像处理
        _, frame = cap1.read()
        frame = rev_cam(frame) # 摄像头倒转添加
        frame = cv2.resize(frame, (width1, height1)) # 缩放比例
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度处理
        #cv2.imshow("grey", frame)
        res = frame[75:115, :] #切割图像的下半部分
        _, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY) # 二值化
        cv2.imshow("review1", res)
        cv2.waitKey(1)
        
        print(time.time()-start_time)
        #qi pan ge
        if (time.time() - start_time) > time3:
            num_white_pixels = cv2.countNonZero(res)
            num_black_pixels = res.size - num_white_pixels
            print("num_black_pixels:", num_black_pixels)
            glcm=greycomatrix(res, distances=[5],angles=[90],levels=256,symmetric=True,normed=True)
            correlation_feature=greycoprops(glcm,'correlation')[0][0]
            print("Correlation Feature:", correlation_feature)
            if num_black_pixels > black_pixels_qipan and correlation_feature < 0.1:
                print("qipan")
                #robot.movement.stop(times=1000)
                break
        
        #模板匹配，得到value
        results = []
        for img in template_imgs:
            results.append(cv2.matchTemplate(res, img, cv2.TM_CCORR_NORMED))
            value = np.argmax(results) # 0:forward 1：left 2:right 3：left1
        
        # 区分黑白
        if micai == 0 and value == 0:
            num_white_pixels = cv2.countNonZero(res)
            num_black_pixels = res.size - num_white_pixels
            print("num_black_pixels:", num_black_pixels)
            if num_black_pixels > black_pixels_micai:
                glcm=greycomatrix(res, distances=[5],angles=[90],levels=256,symmetric=True,normed=True)
                correlation_feature=greycoprops(glcm,'correlation')[0][0]
                print("Correlation Feature:", correlation_feature)
                if isback == 0 and num_black_pixels > black_pixels_back and correlation_feature > 0.5:
                    value = 4
                    isback = 1
                if correlation_feature <= 0.5:
                    micai = 1 #进入迷彩
                    time2 = time.time() - start_time
                    time3 = time2 + 8
                    print("mi cai")
        
        print('img_out:', value)
        #根据value值输出运动
        if value == 0:
            print("forward")
            if (time.time() - start_time) < time2:
                robot.movement.move_forward(speed=250,times=500)
            else:
                robot.movement.move_forward(speed=60,times=200)
        elif value == 1:# or value == 3:
            print("left")
            if (time.time() - start_time) < time2:
                robot.movement.turn_left(angle=300,speed1=7,speed=70,times=200)
                #time.sleep(time_left*0.001)
            elif (time.time() - start_time) < time3:
                robot.movement.turn_left(speed=60,times=200)
            else:
                robot.movement.turn_left(speed=35,times=200)
                #time.sleep(0.15)
        elif value == 2:  # or value == 3:
            print("right")
            if (time.time() - start_time) < time2:
                robot.movement.turn_right(speed=100,times=200)
            elif (time.time() - start_time) < time3:
                robot.movement.turn_right(speed=60,times=200)
            else:
                robot.movement.turn_right(speed=35,times=200)
                #time.sleep(0.15)
        elif value == 4:
            print("back!!!!!!!")
            robot.movement.move_backward(times=1500)
            time.sleep(1.5)
            robot.movement.turn_left(times = 1300)
            time.sleep(1.3)
        elif value == 3 and isback == 0:
            print("left1!!!!!!")
            isback = 1
            robot.movement.move_backward(times=800)
            time.sleep(0.8)
            robot.movement.turn_left(speed=70,times=900)
            time.sleep(0.9)
        elif cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cv2.destroyAllWindows()
    cap2 = cv2.VideoCapture(1)  # 背后的摄像头
    iscenter = 0  # 不在中间
    isclose = 0  # 不够近
    while cap2.isOpened():
        # 处理图像
        _, frame2 = cap2.read()
        frame2 = rev_cam(frame2)
        frame2 = cv2.resize(frame2, (width2, height2))  # 缩放比例
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        res2 = frame2[60:300, :]  # slice the lower part of a frame
        _, res2 = cv2.threshold(res2, threshold2, 255, cv2.THRESH_BINARY)
        cv2.imshow("review2", res2)
        cv2.waitKey(1)

        # find circle
        circles = cv2.HoughCircles(res2, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=100)
        current_x = 0  # 当前圆心坐标
        current_r = 0  # 当前半径大小
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 得到当前圆的参数
            for circle in circles[0, :]:
                center_x = circle[0]
                r = circle[2]
                current_x = center_x
                current_r = r

            # 左右移动
            if current_x < target_x - tolerant_x:
                iscenter = 0
                print("left")
                robot.movement.move_left(times=20)
            elif current_x > target_x + tolerant_x:
                iscenter = 0
                print("right")
                robot.movement.move_right(times=20)
            else:
                iscenter = 1

            # 前进
            if isclose == 0: # and iscenter == 1:
                if current_r < target_r:
                    print("far")
                    robot.movement.move_forward(speed=21,times=50)
                else:
                    print("close")
                    isclose = 1

            # 到中间且距离近，击打
            if iscenter == 1 and isclose == 1:
                robot.movement.hit()
                break

        elif isclose == 0: # 没找到圆且远
            print("no circle")
            robot.movement.move_forward(speed=21,times=50)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    auto_pilot()






