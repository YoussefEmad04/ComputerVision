import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # to change volume of our computer
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to rgb
        self.results = self.hands.process(imgRGB)  # process the frame and give us the results
        if self.results.multi_hand_landmarks:  # check if something detect or not
            for handLms in self.results.multi_hand_landmarks:  # for each hand landmark we will get results
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # handlms should draw hand for us,handconnections draw connections for us
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(
                    myHand.landmark):  # get information on hand the id and the info of coordinates landmarks
                h, w, c = img.shape  # pixel coordinate of circle shape
                cx, cy = int(lm.x * w), int(
                    lm.y * h)  # position of center integer should convert to for width and height
                lmList.append([id, cx, cy])
                if draw and id in self.tipIds:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # draw circle in the centers for fingers
        return lmList


def main():
    pTime = 0
    cTime = 0  # currenttime
    cap = cv2.VideoCapture(0)
    detector = handDetector()  # class above

    # Initialize pycaw for volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    volBar = 400
    volper = 0
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            # Get the coordinates of the thumb (tip) and the index finger (tip)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Calculate the distance between thumb and index finger
            length = math.hypot(x2 - x1, y2 - y1)

            # Map the length to the volume range
            vol = np.interp(length, [50, 300],
                            [minVol, maxVol])  # by numpy.interp we convert the length and get the range to which we map
            volBar = np.interp(length, [50, 300], [400, 150])
            volper = np.interp(length, [50, 300], [0, 100])  # Mapping length to 0% - 100%

            # Set the system volume
            volume.SetMasterVolumeLevel(vol, None)

            # Calculate the midpoint between the thumb and index finger
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Button effect: change color or size when fingers are close
            if length < 50:  # Threshold distance for the "button press"
                cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)  # Green color and larger size
                # Additional effects like sound or vibration could be triggered here if desired
            else:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Original color and size

            # Draw a line between thumb and index finger
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Show the current volume level on the screen
            cv2.putText(img, f'Volume: {int(volper)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 250, 0), 3)

        # Draw the volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # fps 1/currenttime - previous time
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)  # add text fps int and value of position and font and scale and color

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
