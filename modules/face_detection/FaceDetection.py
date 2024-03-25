import cv2
import mediapipe as mp
import time
import os.path


class FaceDetector():
    def __init__(self, minDetectionConfidence=0.5, model_selection=1):
        self.minDetectionConfidence = minDetectionConfidence
        self.model_selection = model_selection

        self.mpFace = mp.solutions.face_detection
        self.faceDetection = self.mpFace.FaceDetection(self.minDetectionConfidence, self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        boundingBoxes = []
        # Bounding box
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                # print(detection.location_data.relative_bounding_box)
                # mpDraw.draw_detection(img, detection)         # Not used. We're drawing our box.
                boundingBox_fromClass = detection.location_data.relative_bounding_box
                height, width, channels = img.shape
                boundingBox = int(boundingBox_fromClass.xmin * width), int(boundingBox_fromClass.ymin * height),    \
                              int(boundingBox_fromClass.width * width), int(boundingBox_fromClass.height * height)
                boundingBoxes.append([id, boundingBox, detection.score])
                if draw:
                    #cv2.rectangle(img, boundingBox, (255, 0, 255), 2)
                    img = self.drawBox(img, boundingBox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0] + 5, boundingBox[1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        return img, boundingBoxes

    def drawBox(self, img, boundingBox, length=40, thickness=2):
        x, y, width, height = boundingBox
        x1, y1 = x + width, y + height
        cv2.rectangle(img, boundingBox, (255, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x+length, y-length//2), (255, 0, 255), thickness)

        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)

        return img

def main():
    currPath = os.path.dirname(__file__)
    rootPath = currPath + "/../../"
    cap = cv2.VideoCapture(rootPath + "videos/face_detection_videos/4.mp4")
    # cap = cv2.VideoCapture(0)

    prevTime = 0

    faceDetector = FaceDetector()       # minDetectionConfidence=0.75, model_selection=1

    while True:
        success, img = cap.read()
        img, boundingBoxes = faceDetector.findFaces(img)
        #print(boundingBoxes)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()