import cv2
import mediapipe as mp
import time
import os.path

def preResize(cap, max_height = 720):
    '''Custom function to keep the video inside some boundaries (720px high by default)'''
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ratio = width / height
    new_width = int(max_height * ratio)
    dim = (new_width, max_height)  # dim = (width, height)
    return dim

currPath = os.path.dirname(__file__)
rootPath = currPath + "/../../"
cap = cv2.VideoCapture(rootPath + "videos/pose_videos/3.mp4")
# dim = preResize(cap)

mpPose = mp.solutions.pose
pose = mpPose.Pose()            # Default parameters: static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                                # enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

prevTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            print(id, cx, cy)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)

    cv2.waitKey(1)