import cv2


def rev_cam(frame):
    (h, w) = frame.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1)  # 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated
