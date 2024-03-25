''' Bare minimum code to run this'''

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Create object hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()      # default parameters : static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # hands only used RGB images
    results = hands.process(imgRGB)

    # Draw hands landmarks from camera image
    if (results.multi_hand_landmarks):
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                '''if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)'''


            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)          # We are displaying the original image, not the RGB image


    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)    # Draw FPS in the image corner

    cv2.imshow("Image", img)
    cv2.waitKey(1)
