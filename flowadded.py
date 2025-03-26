import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Fuel flow rate in liters per second
FUEL_FLOW_RATE_LPS = 0.10

class FuelStationSafetyMonitor:
    def __init__(self, video_path):
        self.nozzle_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/runs/detect/fuel_nozzle_detection/weights/best.pt")
        self.car_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/myenv/Scripts/runs/detect/train/weights/best.pt")
        self.pose_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/yolov8s-pose.pt")
        self.fire_smoke_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/YOLOv8-Fire-and-Smoke-Detection-main/runs/detect/train/weights/best.pt")

        self.alert_status = {'fuel_nozzle': False, 'fire': False, 'smoke': False, 'pose': False, 'fueling': False}
        self.video_path = video_path
        self.total_fuel_injected = 0.0

    def process_frame(self, frame, frame_time):
        nozzle_results = self.nozzle_model(frame)
        car_results = self.car_model(frame)
        pose_results = self.pose_model(frame)
        fire_smoke_results = self.fire_smoke_model(frame)

        processed_frame = nozzle_results[0].plot()

        nozzle_boxes = [box.xyxy.cpu().numpy() for box in nozzle_results[0].boxes]
        car_boxes = [box.xyxy.cpu().numpy() for box in car_results[0].boxes]
        fire_smoke_boxes = [box.xyxy.cpu().numpy() for box in fire_smoke_results[0].boxes]

        self.alert_status['fuel_nozzle'] = len(nozzle_boxes) > 0
        self.alert_status['fueling'] = False

        for nozzle_box in nozzle_boxes:
            for car_box in car_boxes:
                if self.boxes_touch(nozzle_box[0], car_box[0]):
                    self.alert_status['fueling'] = True
                    fuel_injected = FUEL_FLOW_RATE_LPS * frame_time
                    self.total_fuel_injected += fuel_injected
                    print(f"Fueling detected! Fuel injected: {fuel_injected:.2f} L, Total: {self.total_fuel_injected:.2f} L")

        self.alert_status['fire'] = False
        self.alert_status['smoke'] = False
        for box in fire_smoke_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls.cpu().numpy())

            if class_id == 0:
                self.alert_status['fire'] = True
                color = (0, 165, 255)
                label = "Fire"
            elif class_id == 1:
                self.alert_status['smoke'] = True
                color = (192, 192, 192)
                label = "Smoke"
            else:
                continue

            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for car_box in car_results[0].boxes:
            x1, y1, x2, y2 = map(int, car_box.xyxy[0].cpu().numpy())
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(processed_frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"Detected {len(car_results[0].boxes)} cars, {len(nozzle_results[0].boxes)} fuel nozzles, {len(fire_smoke_results[0].boxes)} fire/smoke.")

        if pose_results:
            for result in pose_results:
                for keypoint in result.keypoints.xy.cpu().numpy():
                    for x, y in keypoint:
                        cv2.circle(processed_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            self.alert_status['pose'] = True
        else:
            self.alert_status['pose'] = False

        self.trigger_alerts()
        return processed_frame

    def boxes_touch(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    def trigger_alerts(self):
        alerts = [k for k, v in self.alert_status.items() if v]
        if alerts:
            print(f"ALERT: {', '.join(alerts)} detected!")

def main(video_path):
    monitor = FuelStationSafetyMonitor(video_path)
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    prev_time = cv2.getTickCount()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        current_time = cv2.getTickCount()
        frame_time = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time

        processed_frame = monitor.process_frame(frame, frame_time)
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved as output_video.mp4\nTotal Fuel Injected: {monitor.total_fuel_injected:.2f} L")

if __name__ == "__main__":
    video_path = "C:/Users/pbani/Downloads/videoplayback (1).mp4"
    main(video_path)