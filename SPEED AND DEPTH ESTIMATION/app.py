import cv2
import numpy as np
from tracker import EuclideanDistTracker

# Configuration Section
VIDEO_PATH = "video.mp4"
FRAME_SKIP = 10
KERNEL_OP = np.ones((3, 3), np.uint8)
KERNEL_CL = np.ones((11, 11), np.uint8)
KERNEL_E = np.ones((5, 5), np.uint8)
THRESHOLD_AREA = 1000

# Create Tracker Object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture(VIDEO_PATH)

# Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    height, width, _ = frame.shape
    roi = frame[20:540, 100:960]

    # Masking
    fgmask = object_detector.apply(roi)
    _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    depth_map_color = cv2.applyColorMap(mask, cv2.COLORMAP_PLASMA)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OP)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CL)
    e_img = cv2.erode(mask, KERNEL_E)

    contours, _ = cv2.findContours(e_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > THRESHOLD_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        speed = tracker.get_speed(id)

        if speed < tracker.limit():
            color = (0, 255, 0)
        else:
            color = (199, 44, 72)

        cv2.putText(roi, f"ID: {id} Speed: {speed} km/h", (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), color, 3)

        if tracker.should_capture(id):
            tracker.capture(roi, x, y, h, w, speed, id)

    # Draw Lines
    cv2.line(roi, (0, 410), (960, 60), (199, 44, 72), 2)
    cv2.line(roi, (0, 430), (960, 90), (199, 44, 72), 2)
    cv2.line(roi, (0, 235), (960, 15), (199, 44, 72), 2)
    cv2.line(roi, (0, 255), (960, 25), (199, 44, 72), 2)

    # Display
    cv2.imshow("DEPTH", depth_map_color)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(FRAME_SKIP)
    if key == 27:
        tracker.end()
        break

tracker.end()
cap.release()
cv2.destroyAllWindows()