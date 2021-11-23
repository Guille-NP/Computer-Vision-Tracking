import cv2
import time
import PoseModule as pm


cap = cv2.VideoCapture('PoseVideos/6.mp4')
dim = pm.preResize(cap)  # Resizing Video dimension for cv2.resize(...)
pTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = cv2.resize(img, dim,
                     interpolation=cv2.INTER_AREA)  # Resize video
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList)
    cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 255, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)