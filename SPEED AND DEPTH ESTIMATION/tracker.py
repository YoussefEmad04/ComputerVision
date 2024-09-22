import cv2
import math
import time
import numpy as np
import os
import random

# Constants
LIMIT_SPEED = 80  # km/hr
DISTANCE_BETWEEN_LINES = 10  # meters, example value; use actual distance
TRAFFIC_RECORD_FOLDER = "TrafficRecord"
EXCEEDED_FOLDER = os.path.join(TRAFFIC_RECORD_FOLDER, "exceeded")
SPEED_RECORD_FILE = os.path.join(TRAFFIC_RECORD_FOLDER, "SpeedRecord.txt")

# Ensure directories exist
os.makedirs(EXCEEDED_FOLDER, exist_ok=True)

# Initialize speed record file
with open(SPEED_RECORD_FILE, "w") as file:
    file.write("ID\tSPEED\n------\t-------\n")

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.s1 = np.zeros(1000)
        self.s2 = np.zeros(1000)
        self.s = np.zeros(1000)
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx, cy = (x + w // 2), (y + h // 2)
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True

                    # Start timer when vehicle crosses the first line
                    if 410 <= y <= 430:
                        if self.s1[id] == 0:
                            self.s1[id] = time.time()
                    # Stop timer when vehicle crosses the second line
                    if 235 <= y <= 255:
                        if self.s1[id] != 0 and self.s2[id] == 0:
                            self.s2[id] = time.time()
                            self.s[id] = self.s2[id] - self.s1[id]
                            self.f[id] = 1  # Mark as crossed both lines


            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        self.center_points = {obj_bb_id[4]: self.center_points[obj_bb_id[4]] for obj_bb_id in objects_bbs_ids}
        return objects_bbs_ids

    def get_speed(self, id):
        self.s[id] += random.random()
        if self.s[id] != 0:
            speed_mps = DISTANCE_BETWEEN_LINES / self.s[id]
            speed_kmph = speed_mps * 5.6  # Convert m/s to km/h
        else:
            speed_kmph = 0
        return int(speed_kmph)

    def should_capture(self, id):
        return self.f[id] == 1 and self.s[id] != 0

    def capture(self, img, x, y, h, w, sp, id):
        if self.capf[id] == 0:
            self.capf[id] = 1
            self.f[id] = 0
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            n = f"{id}_speed_{sp}"
            file_path = os.path.join(TRAFFIC_RECORD_FOLDER, f"{n}.jpg")
            cv2.imwrite(file_path, crop_img)
            self.count += 1
            with open(SPEED_RECORD_FILE, "a") as file:
                if sp > LIMIT_SPEED:
                    exceeded_path = os.path.join(EXCEEDED_FOLDER, f"{n}.jpg")
                    cv2.imwrite(exceeded_path, crop_img)
                    file.write(f"{id}\t{sp}<---exceeded\n")
                    self.exceeded += 1
                else:
                    file.write(f"{id}\t{sp}\n")

    def limit(self):
        return LIMIT_SPEED

    def end(self):
        with open(SPEED_RECORD_FILE, "a") as file:
            file.write("\n-------------\n")
            file.write("SUMMARY\n")
            file.write("-------------\n")
            file.write(f"Total Vehicles:\t{self.count}\n")
            file.write(f"Exceeded speed limit:\t{self.exceeded}\n")