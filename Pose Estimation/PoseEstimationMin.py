import cv2
import mediapipe as mp
import time

def preResize(cap, max_height = 720):
    '''Custom function to keep the video inside some boundaries (720px high by default)'''
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ratio = width / height
    new_width = int(max_height * ratio)
    dim = (new_width, max_height)  # dim = (width, height)
    return dim

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,  # Default parameters
                   enable_segmentation=False, smooth_segmentation=True,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('PoseVideos/3.mp4')

dim = preResize(cap)       # Resizing Video dimension for cv2.resize(...) (Custom function, see above)

pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  # Resize video (dim should be different for each video)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color codification so openCV and Mediapipe are compatible
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
