import cv2
import time
import PoseEstimationMod as pm
import os.path

# cap = cv2.VideoCapture(0)
currPath = os.path.dirname(__file__)
rootPath = currPath + "/../../"
cap = cv2.VideoCapture(rootPath + "videos/pose_videos/4.mp4")
# dim = preResize(cap)
prevTime = 0

estimator = pm.poseEstimator()

while True:
    success, img = cap.read()
    img = estimator.findPose(img)
    lmList = estimator.getPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)  # lmList[4] for a specific landmark
        # Highlight a specific LM:
        landmark = 15
        cv2.circle(img, (lmList[landmark][1], lmList[landmark][2]), 5, (255, 0, 0), cv2.FILLED)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)

    cv2.waitKey(1)