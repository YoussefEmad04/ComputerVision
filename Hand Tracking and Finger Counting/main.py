import cv2
import mediapipe as mp
import time

class handDetector():
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
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to rgb
        self.results = self.hands.process(imgRGB) #process the frame and give us the results
        if self.results.multi_hand_landmarks: #chcek if something detect or not
            for handLms in self.results.multi_hand_landmarks: #for each hand land mark we will get results
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)#handlms should draw hand for us,hnadconnections draw connections for us
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):#get infromation on hand the id and the info of cordinatess landmarks
                h, w, c = img.shape #pixel cordinate of circle shape
                cx, cy = int(lm.x * w), int(lm.y * h) #postion of center integer should convert to for width and height
                lmList.append([id, cx, cy])
                if draw and id in self.tipIds:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) #hna hn3ml draw circle fe el centers llfingers
        return lmList

    def fingersUp(self, lmList):
        fingers = []

        # Thumb is the x corrdinate
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers (Index, Middle, Ring, Pinky) are the y corrdinate
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0 #currenttime
    cap = cv2.VideoCapture(0)
    detector = handDetector() #eli fe el class fo2

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = []
        if detector.results.multi_hand_landmarks:
            for handNo in range(len(detector.results.multi_hand_landmarks)):
                lmList.extend(detector.findPosition(img, handNo))

        if lmList:
            totalFingers = 0 #hna bylf 3al fingers w by3edohm 2 byrg3li el totol zy counter
            for handNo in range(len(detector.results.multi_hand_landmarks)):
                fingers = detector.fingersUp(lmList)
                totalFingers += fingers.count(1)

            cv2.putText(img, f'Total Fingers: {totalFingers}', (20, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) #hna byput text brdo bs totla fingers

        cTime = time.time()
        fps = 1 / (cTime - pTime) #the fbs 1/currenttime - previous time
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2)#hmn7ot hna el text ka fbs int w value llpoition w el font w el scale w el colour

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
