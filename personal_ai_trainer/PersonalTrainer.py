import cv2
import numpy as np
from modules.pose_estimation import PoseEstimation as pe
import os.path

currPath = os.path.dirname(__file__)
cap = cv2.VideoCapture(currPath + "/videos/1.mp4")
# cap = cv2.VideoCapture(0)


detector =pe.poseEstimator()

reps_count = 0
lifting = 1

while True:
    success, img = cap.read()

    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)

    if lmList:
        # Right arm
        angle_right = detector.getAngle(img, 13, 11, 15)  # elbow, shoulder and wrist
        # Left arm
        angle_left = detector.getAngle(img, 14, 12, 16)  # elbow, shoulder and wrist
        angle = angle_right
        if angle < 25 and lifting:
            reps_count += 1
            lifting = 0
        elif angle > 155 and ~lifting:
            lifting = 1

        cv2.putText(img, str(int(reps_count)), (70, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
        cv2.putText(img, str(int(angle)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)