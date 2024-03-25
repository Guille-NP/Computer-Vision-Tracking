import cv2
import mediapipe as mp
import time
import os.path

currPath = os.path.dirname(__file__)
rootPath = currPath + "/../../"
cap = cv2.VideoCapture(rootPath + "videos/face_detection_videos/4.mp4")
#cap = cv2.VideoCapture(0)

prevTime = 0

mpFace = mp.solutions.face_detection
faceDetection = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)

    # Bounding box
    if result.detections:
        for id, detection in enumerate(result.detections):
            # print(detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img, detection)         # Not used. We're drawing our box.
            boundingBox_fromClass = detection.location_data.relative_bounding_box
            height, width, channels = img.shape
            boundingBox = int(boundingBox_fromClass.xmin * width), int(boundingBox_fromClass.ymin * height),    \
                          int(boundingBox_fromClass.width * width), int(boundingBox_fromClass.height * height)
            cv2.rectangle(img, boundingBox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] + 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    currTime = time.time()
    fps = 1 /(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
