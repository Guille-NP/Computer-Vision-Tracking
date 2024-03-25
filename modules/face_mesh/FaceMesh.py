import cv2
import mediapipe as mp
import time
import os.path


class FaceMesh():
    def __init__(self, mode=False, max_num_faces=5, refine_landmarks=False, min_detection_confidence=0.75, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.max_num_faces, self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findLandmarks(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(imgRGB)
        height, width, channels = img.shape
        # Landmarks representation
        facesList = []
        if self.result.multi_face_landmarks:
            for i, faceLandmarks in enumerate(self.result.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpecs, self.drawSpecs)
                landmarksList = []
                for id, landmark in enumerate(faceLandmarks.landmark):
                    coord_x, coord_y = int(landmark.x * width), int(landmark.y * height)
                    landmarksList.append([id, coord_x, coord_y])
                facesList.append(landmarksList)
        return facesList


def main():
    currPath = os.path.dirname(__file__)
    rootPath = currPath + "/../../"
    cap = cv2.VideoCapture(rootPath + "videos/face_mesh_videos/4.mp4")
    # cap = cv2.VideoCapture(0)
    prevTime = 0

    faceMesh = FaceMesh()

    while True:
        success, img = cap.read()

        facesList = faceMesh.findLandmarks(img, draw=True)
        # if len(facesList) != 0:
        #     print(facesList)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()