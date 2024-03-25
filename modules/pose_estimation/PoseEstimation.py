import cv2
import mediapipe as mp
import time
import os.path
import math

def preResize(cap, max_height = 720):
    '''Custom function to keep the video inside some boundaries (720px high by default)'''
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ratio = width / height
    new_width = int(max_height * ratio)
    dim = (new_width, max_height)  # dim = (width, height)
    return dim

class poseEstimator():
    def __init__(self, mode=False, complexity=1, upper_body=False, smooth_landmarks=True, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.upper_body = upper_body
        self.smooth_landmarks = smooth_landmarks
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upper_body, self.smooth_landmarks, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def getAngle(self, img, center, p1, p2, draw=True):
        center_x, center_y = self.lmList[center][1:]
        p1_x, p1_y = self.lmList[p1][1:]
        p2_x, p2_y = self.lmList[p2][1:]

        subAngle1 = 180 - math.degrees(math.atan2(center_y - p1_y, center_x - p1_x))
        subAngle2 = math.degrees(math.atan2(center_y - p2_y, center_x - p2_x))
        angle = 180 - subAngle1 - subAngle2
        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (p1_x, p1_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (p2_x, p2_y), 10, (255, 0, 0), cv2.FILLED)

        return angle



def main():
    # cap = cv2.VideoCapture(0)
    currPath = os.path.dirname(__file__)
    rootPath = currPath + "/../../"
    cap = cv2.VideoCapture(rootPath + "videos/pose_videos/3.mp4")
    # dim = preResize(cap)
    prevTime = 0

    estimator = poseEstimator()

    while True:
        success, img = cap.read()
        img = estimator.findPose(img)
        lmList = estimator.getPosition(img)

        if len(lmList) != 0:
            # print(lmList)   # lmList[4] for a specific landmark
            # Highlight a specific LM:
            landmark = 12
            cv2.circle(img, (lmList[landmark][1], lmList[landmark][2]), 10, (255, 0, 0), cv2.FILLED)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()