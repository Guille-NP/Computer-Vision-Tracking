import cv2
import time
import numpy as np
import math

from modules.hand_tracking import HandTracking as ht


# Web-cam capture
capWidth = 1280  # capWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # ---> 640 (default)
capHeight = 720  # capHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # ---> 480 (default)
cap = cv2.VideoCapture(0)
cap.set(3, capWidth)
cap.set(4, capHeight)

# Hand detector instance
hand_detector = ht.handDetector(detectionConfidence=0.75)
fingerTips = [8, 12, 16, 20]
lm_color = (255, 0, 0)

# FPS initalisation
prevTime = 0

while True:
    success, img = cap.read()

    hand_detector.findHands(img)
    landmarkList = hand_detector.findPostion(img, draw=False)

    if len(landmarkList) != 0:
        fingersUp = []
        # Check if right or left hand
        left_hand = False
        right_hand = False
        if landmarkList[8][1] < landmarkList[20][1]:
            left_hand = True
            right_hand = False
            cv2.putText(img, "Left Hand", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        elif landmarkList[8][1] > landmarkList[12][1]:
            left_hand = False
            right_hand = True
            cv2.putText(img, "Right Hand", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            left_hand = False
            right_hand = False

        # Thumb
        if left_hand:
            if landmarkList[4][1] < landmarkList[3][1]:
                fingersUp.append(1)
                cv2.circle(img, (landmarkList[4][1], landmarkList[4][2]), 10, lm_color, cv2.FILLED)
            else:
                fingersUp.append(0)
        elif right_hand:
            if landmarkList[4][1] > landmarkList[3][1]:
                fingersUp.append(1)
                cv2.circle(img, (landmarkList[4][1], landmarkList[4][2]), 10, lm_color, cv2.FILLED)
            else:
                fingersUp.append(0)

        # Fingers
        for fingerTip in fingerTips:
            if landmarkList[fingerTip][2] < landmarkList[fingerTip - 2][2]:
                fingersUp.append(1)
                cv2.circle(img, (landmarkList[fingerTip][1], landmarkList[fingerTip][2]), 10, lm_color, cv2.FILLED)
            else:
                fingersUp.append(0)

        cv2.putText(img, str(int(fingersUp.count(1))), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Draw FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
