import mediapipe as mp
import math, cv2


class GestureDetector:
    def __init__(self, display_result=True):
        self.__display_result = display_result
        self.__draw = mp.solutions.drawing_utils
        self.__hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
        self.window_name = "MediaPipe Hands"

    def process(self, frame):
        frame = cv2.resize(frame, (640, 480))

        hands_landmarks = self.__find_hind(frame)

        result_number = None

        if hands_landmarks:
            result_number = self.__detect_number(hands_landmarks, frame)

            if self.__display_result:

                if result_number >= 0:
                    cv2.putText(frame, str(result_number), (150, 150), 19, 5, (255, 0, 255), 5, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "NO NUMBER", (150, 150), 20, 1, (0, 0, 255))

                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)

        return result_number

    def __find_hind(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        hand_lms_style = self.__draw.DrawingSpec(color=(0, 0, 255), thickness=5)
        hand_con_style = self.__draw.DrawingSpec(color=(0, 255, 0), thickness=5)

        results = self.__hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.__draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS, hand_lms_style,
                                           hand_con_style)
        return results.multi_hand_landmarks

    def __detect_number(self, hand_landmarks, img):

        h, w, c = img.shape

        myhand = hand_landmarks[0]
        hand_landmark = myhand.landmark
        thumb_tip_id = 4  # 大拇指指尖
        index_finger_tip_id = 8  # 食指指尖
        middle_finger_tip_id = 12  # 中指指尖
        ring_finger_tip_id = 16  # 无名指指尖
        pinky_finger_tip_id = 20  # 小指指尖
        pinky_finger_mcp_id = 17  # 小指指根（用于判断4和5）
        wrist_id = 0  # 手腕（用于识别数字6）

        # 提取y坐标
        thumb_tip_y = hand_landmark[thumb_tip_id].y * h
        index_tip_y = hand_landmark[index_finger_tip_id].y * h
        middle_tip_y = hand_landmark[middle_finger_tip_id].y * h
        ring_tip_y = hand_landmark[ring_finger_tip_id].y * h
        pinky_tip_y = hand_landmark[pinky_finger_tip_id].y * h
        pinky_mcp_y = hand_landmark[pinky_finger_mcp_id].y * h
        wrist_y = hand_landmark[wrist_id].y * h

        # 提取x坐标
        thumb_tip_x = hand_landmark[thumb_tip_id].x * w
        index_tip_x = hand_landmark[index_finger_tip_id].x * w
        middle_tip_x = hand_landmark[middle_finger_tip_id].x * w
        ring_tip_x = hand_landmark[ring_finger_tip_id].x * w
        pinky_tip_x = hand_landmark[pinky_finger_tip_id].x * w
        pinky_mcp_x = hand_landmark[pinky_finger_mcp_id].x * w
        wrist_x = hand_landmark[wrist_id].x * w

        dist_thumb2wrist = math.sqrt((thumb_tip_x - wrist_x) ** 2 + (thumb_tip_y - wrist_y) ** 2)
        dist_index2wrist = math.sqrt((index_tip_x - wrist_x) ** 2 + (index_tip_y - wrist_y) ** 2)
        dist_middle2wrist = math.sqrt((middle_tip_x - wrist_x) ** 2 + (middle_tip_y - wrist_y) ** 2)
        dist_ring2wrist = math.sqrt((ring_tip_x - wrist_x) ** 2 + (ring_tip_y - wrist_y) ** 2)
        dist_pinky2wrist = math.sqrt((pinky_tip_x - wrist_x) ** 2 + (pinky_tip_y - wrist_y) ** 2)
        dist_pinky_mcp2wrist = math.sqrt((thumb_tip_x - pinky_mcp_x) ** 2 + (thumb_tip_y - pinky_mcp_y) ** 2)

        # 相当于取dist_thumb2wrist_ratio == 1
        dist_index2wrist_ratio = dist_index2wrist / dist_thumb2wrist
        dist_middle2wrist_ratio = dist_middle2wrist / dist_thumb2wrist
        dist_ring2wrist_ratio = dist_ring2wrist / dist_thumb2wrist
        dist_pinky2wrist_ratio = dist_pinky2wrist / dist_thumb2wrist
        dist_pinky_mcp2wrist_ratio = dist_pinky_mcp2wrist / dist_thumb2wrist

        # print(dist_index2wrist_ratio, dist_middle2wrist_ratio, dist_ring2wrist_ratio, dist_pinky2wrist_ratio, dist_pinky_mcp2wrist_ratio)

        if dist_index2wrist_ratio < 1.9 and dist_middle2wrist_ratio < 1.8 and dist_ring2wrist_ratio < 1.6 and dist_pinky2wrist_ratio < 1.4 and dist_pinky_mcp2wrist_ratio < 0.8:
            return 0
        elif 2.0 < dist_index2wrist_ratio and dist_middle2wrist_ratio < 1.8 and dist_ring2wrist_ratio < 1.6 and dist_pinky2wrist_ratio < 1.4 and dist_pinky_mcp2wrist_ratio < 0.8:
            return 1
        elif 2.0 < dist_index2wrist_ratio and 1.9 < dist_middle2wrist_ratio and dist_ring2wrist_ratio < 1.6 and dist_pinky2wrist_ratio < 1.4 and dist_pinky_mcp2wrist_ratio < 0.8:
            return 2
        elif 2.0 < dist_index2wrist_ratio and 1.9 < dist_middle2wrist_ratio and 1.75 < dist_ring2wrist_ratio and dist_pinky2wrist_ratio < 1.4 and dist_pinky_mcp2wrist_ratio < 0.8:
            return 3
        elif 2.0 < dist_index2wrist_ratio and 1.9 < dist_middle2wrist_ratio and 1.75 < dist_ring2wrist_ratio and 1.5 < dist_pinky2wrist_ratio and dist_pinky_mcp2wrist_ratio < 0.8:
            return 4
        elif dist_index2wrist_ratio > 0.5 and dist_middle2wrist_ratio > 0.5 and dist_ring2wrist_ratio > 0.5 and 0.9 < dist_pinky_mcp2wrist_ratio < 1.2:
            return 5
        elif dist_index2wrist_ratio < 0.5 and dist_middle2wrist_ratio < 0.5 and dist_ring2wrist_ratio < 0.5:
            return 6
        else:
            return -1
