import cv2
import time
import numpy as np
import math

from modules.hand_tracking import HandTracking as ht
from utilities import VolumeDriver


# Web-cam capture
capWidth = 1280  # capWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # ---> 640 (default)
capHeight = 720  # capHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # ---> 480 (default)
cap = cv2.VideoCapture(0)
cap.set(3, capWidth)
cap.set(4, capHeight)

# Hand detector instance
hand_detector = ht.handDetector(detectionConfidence=0.75)

# Volume driver instance and graphics
volume_driver = VolumeDriver()
volume_range = volume_driver.GetVolumeRange()
minVolume_dB, maxVolume_dB = volume_range[0], volume_range[1]
volume = volume_driver.GetMasterVolumeLevel()

volume_percentage = 0       # TO-DO: Calculate percentage from volume (dB --> %)
volume_bar = 300            # TO-DO: Initiate bar according to percentage

# FPS initalisation
prevTime = 0

while True:
    success, img = cap.read()

    hand_detector.findHands(img)
    landmarkList = hand_detector.findPostion(img, draw=False)

    if len(landmarkList) != 0:
        x0, y0 = landmarkList[4][1], landmarkList[4][2]      # Thumb
        x1, y1 = landmarkList[8][1], landmarkList[8][2]      # Index
        fingerTip0, fingerTip1 = (x0, y0), (x1, y1)
        distance = math.hypot(x0 - x1, y0 - y1)
        if distance < 30:    lm_color = (0, 255, 0)
        elif distance > 300: lm_color = (0, 0, 255)
        else:                lm_color = (255, 255, 0)

        # Volume control: Fingertips distance range ---> 30 - 300
        volume = np.interp(distance, [30, 300], [minVolume_dB, maxVolume_dB])   # TO-DO: Assuming it is linear.
        volume_percentage = np.interp(distance, [30, 300], [0, 100])            # Change to logarithmic
        volume_bar = int(np.interp(distance, [30, 300], [300, 100]))
        volume_driver.SetMasterVolumeLevel(volume)

        # Draw fingertips
        cv2.circle(img, fingerTip0, 10, lm_color, cv2.FILLED)
        cv2.circle(img, fingerTip1, 10, lm_color, cv2.FILLED)
        cv2.line(img, fingerTip0, fingerTip1, lm_color, 3)

    # Draw volume bar (TO-DO: Change linear to logarithmic scale (dB))
    cv2.rectangle(img, (30, 100), (80, 300), (0, 255, 0), 3)
    cv2.rectangle(img, (30, volume_bar), (80, 300), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{str(int(volume_percentage))}%', (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
