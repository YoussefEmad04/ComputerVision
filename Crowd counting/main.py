import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import matplotlib.pyplot as plt
import io

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class_names_goal = ['car', 'motorcycle']

model = YOLO('yolov8m')

video = cv2.VideoCapture('traffic.mp4')

width = 1280
height = 720

# Create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4', fourcc, 30.0, (width, height))


# Create mask
def create_mask(width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[200, 400], [1080, 400], [1280, 720], [0, 720]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


mask = create_mask(width, height)

# Define lines for each compartment
line_y = 472
line_x1, line_x2, line_x3, line_x4 = 256, 500, 672, 904

vehicle_count = {
    'left': 0,
    'middle': 0,
    'right': 0
}

counted_vehicle_ids = {
    'left': set(),
    'middle': set(),
    'right': set()
}

up_img = cv2.imread('up.png', cv2.IMREAD_UNCHANGED)
up_img = cv2.resize(up_img, (50, 50))
down_img = cv2.imread('down.png', cv2.IMREAD_UNCHANGED)
down_img = cv2.resize(down_img, (50, 50))


# Convert images to BGR format
def convert_to_bgr(image):
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


up_img = convert_to_bgr(up_img)
down_img = convert_to_bgr(down_img)


# Function to create pie chart
def create_pie_chart(vehicle_count):
    colors = [(219 / 255, 0 / 255, 115 / 255), (191 / 255, 48 / 255, 48 / 255), (0 / 255, 255 / 255, 127 / 255)]

    plt.figure(figsize=(10, 6))
    lanes = list(vehicle_count.keys())

    # Use total vehicle counts for each lane
    total_counts = [vehicle_count[lane] for lane in lanes]

    labels = [f'{lane.upper()}: {total_counts[i]}' for i, lane in enumerate(lanes)]
    sizes = total_counts

    # Ensure sizes do not have zero or NaN values
    if sum(sizes) == 0:
        sizes = [1 for _ in sizes]

    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 18, "fontweight": "bold", "fontfamily": "monospace"}
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Vehicle Count Distribution', fontdict={"fontsize": 20, "fontfamily": "monospace"})

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# Simple tracker
class SimpleTracker:
    def __init__(self, max_age=20):
        self.next_id = 1
        self.tracked_objects = {}
        self.max_age = max_age

    def update(self, detections):
        new_tracked_objects = {}

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            best_match = None
            min_distance = float('inf')

            for obj_id, obj in self.tracked_objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(obj['center']))
                if dist < min_distance:
                    min_distance = dist
                    best_match = obj_id

            if best_match is not None and min_distance < 50:  # Threshold for matching
                new_tracked_objects[best_match] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class': cls,
                    'age': 0
                }
            else:
                new_tracked_objects[self.next_id] = {
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class': cls,
                    'age': 0
                }
                self.next_id += 1

        # Update age for existing objects
        for obj_id in self.tracked_objects:
            if obj_id not in new_tracked_objects:
                self.tracked_objects[obj_id]['age'] += 1
                if self.tracked_objects[obj_id]['age'] < self.max_age:
                    new_tracked_objects[obj_id] = self.tracked_objects[obj_id]

        self.tracked_objects = new_tracked_objects
        return [(obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], obj_id, obj['class'])
                for obj_id, obj in self.tracked_objects.items()]


tracker = SimpleTracker(max_age=20)

while True:
    success, frame = video.read()
    if not success:
        break

    frame = cv2.resize(frame, (width, height))

    # Apply the mask
    image_region = cv2.bitwise_and(frame, frame, mask=mask)

    results = model(image_region, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            class_name = class_names[int(box.cls[0])]
            if class_name not in class_names_goal:
                continue
            confidence = round(float(box.conf[0]) * 100, 2)
            if confidence < 30:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2, float(box.conf[0]), class_name])

    # Draw lines for each compartment
    cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (0, 0, 255), 2)
    cv2.line(frame, (line_x2, line_y), (line_x3, line_y), (0, 0, 255), 2)
    cv2.line(frame, (line_x3, line_y), (line_x4, line_y), (0, 0, 255), 2)

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id, cls = obj
        confidence_pos_x1 = max(0, x1)
        confidence_pos_y1 = max(36, y1)

        cvzone.putTextRect(frame, f'ID: {obj_id} {cls}', (confidence_pos_x1, confidence_pos_y1), 1, 1)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        if line_y - 10 < center_y < line_y + 10:
            if line_x1 < center_x < line_x2:
                if obj_id not in counted_vehicle_ids['left']:
                    counted_vehicle_ids['left'].add(obj_id)
                    vehicle_count['left'] += 1
                    cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (0, 255, 0), 2)
            elif line_x2 < center_x < line_x3:
                if obj_id not in counted_vehicle_ids['middle']:
                    counted_vehicle_ids['middle'].add(obj_id)
                    vehicle_count['middle'] += 1
                    cv2.line(frame, (line_x2, line_y), (line_x3, line_y), (0, 255, 0), 2)
            elif line_x3 < center_x < line_x4:
                if obj_id not in counted_vehicle_ids['right']:
                    counted_vehicle_ids['right'].add(obj_id)
                    vehicle_count['right'] += 1
                    cv2.line(frame, (line_x3, line_y), (line_x4, line_y), (0, 255, 0), 2)

    cvzone.putTextRect(frame, f'At Left : {vehicle_count["left"]}', (50, 50), 2, 2, offset=20, border=2,
                       colorR=(127, 0, 255), colorB=(127, 0, 255))
    cvzone.putTextRect(frame, f'At Middle : {vehicle_count["middle"]}', (500, 50), 2, 2, offset=20, border=2,
                       colorR=(127, 0, 255), colorB=(127, 0, 255))
    cvzone.putTextRect(frame, f'At Right : {vehicle_count["right"]}', (1000, 50), 2, 2, offset=20, border=2,
                       colorR=(127, 0, 255), colorB=(127, 0, 255))

    frame[20:70, 270:320] = down_img
    frame[20:70, 750:800] = up_img
    frame[20:70, 920:970] = up_img

    # Create and overlay bar graph
    graph = create_pie_chart(vehicle_count)
    graph = cv2.resize(graph, (480, 320))
    frame[100:420, 800:1280] = graph

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()