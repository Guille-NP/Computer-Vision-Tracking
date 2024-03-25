import cv2
import mediapipe as mp
import time
import os.path

currPath = os.path.dirname(__file__)
rootPath = currPath + "/../../"
cap = cv2.VideoCapture(rootPath + "videos/face_mesh_videos/4.mp4")
# cap = cv2.VideoCapture(0)

prevTime = 0

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.75)      # static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
mpDraw = mp.solutions.drawing_utils
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    height, width, channels = img.shape
    # Landmarks representation
    if result.multi_face_landmarks:
        for faceLandmarks in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpecs, drawSpecs)
            for id, landmark in enumerate(faceLandmarks.landmark):
                coord_x, coord_y = int(landmark.x * width), int(landmark.y * height)
                print(id, coord_x, coord_y)


    currTime = time.time()
    fps = 1 /(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


