# Minimum code for handtracking with mediapipe

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Related to the hand detecting module
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 0.5, 0.5)       # Default parameters
mpDraw = mp.solutions.drawing_utils             # To draw hand landmarks later

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)             #
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:        # handLms is a single hand
            for id, lm in enumerate(handLms.landmark):      # Track landmarks in particular (from 0 to 20 landmarks, and each one has x, y and z)
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)       # Converts x and y coordinate from decimals (ratio) to pixels
                print(id, cx, cy)
                if id == 4:                                 # Make bigger the landmark labelled with id = 4  (tip of the thumb)
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   # Draws the landmarks and the connections

    # FPS calculation
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)