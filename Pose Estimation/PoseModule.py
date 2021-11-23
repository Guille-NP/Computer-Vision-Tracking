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


class poseDetector():

    def __init__(self, mode=False, modelComplex=1, smoothLm=True, enableSeg=False, smoothSeg=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.modelCom = modelComplex
        self.smoothLm = smoothLm
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelCom, self.smoothLm, self.enableSeg, self.smoothSeg, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color codification so openCV and Mediapipe are compatible
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    dim = preResize(cap)  # Resizing Video dimension for cv2.resize(...)
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = cv2.resize(img, dim,
                         interpolation=cv2.INTER_AREA)  # Resize video
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
        # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 255, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()