import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Fuel flow rate (liters per second) - adjust as per real data
FUEL_FLOW_RATE_LPS = 0.05  # Example: 0.05 liters per second

def detect_fire_smoke(frame):
    """Detects fire and smoke using color-based thresholding and edge detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire detection (orange-red range)
    lower_fire = np.array([0, 100, 100])
    upper_fire = np.array([20, 255, 255])
    mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Smoke detection (grayish-white range)
    lower_smoke = np.array([0, 0, 200])
    upper_smoke = np.array([180, 50, 255])
    mask_smoke = cv2.inRange(hsv, lower_smoke, upper_smoke)
    
    fire_detected = np.any(mask_fire > 0)
    smoke_detected = np.any(mask_smoke > 0)
    
    return fire_detected, smoke_detected

class FuelStationSafetyMonitor:
    def __init__(self, video_path):
        self.nozzle_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/runs/detect/fuel_nozzle_detection/weights/best.pt")  # Nozzle model
        self.car_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/myenv/Scripts/runs/detect/train/weights/best.pt")  # Car detection model
        self.pose_model = YOLO("C:/Users/pbani/Downloads/ACEhackathon/yolov8s-pose.pt")  # Pose detection
        
        self.alert_status = {'fuel_nozzle': False, 'fire': False, 'smoke': False, 'pose': False, 'fueling': False}
        self.video_path = video_path
        self.total_fuel_injected = 0.0  # Track fuel injected

    def process_frame(self, frame, frame_time):
        nozzle_results = self.nozzle_model(frame)
        car_results = self.car_model(frame)
        pose_results = self.pose_model(frame)

        processed_frame = nozzle_results[0].plot()  # Draw bounding boxes for nozzles

        # Get detected bounding boxes
        nozzle_boxes = [box.xyxy.cpu().numpy() for box in nozzle_results[0].boxes]
        car_boxes = [box.xyxy.cpu().numpy() for box in car_results[0].boxes]

        self.alert_status['fuel_nozzle'] = len(nozzle_boxes) > 0
        self.alert_status['fueling'] = False  # Reset fueling status

        # Check if nozzle is inserted in a car
        for nozzle_box in nozzle_boxes:
            for car_box in car_boxes:
                if self.boxes_touch(nozzle_box[0], car_box[0]):
                    self.alert_status['fueling'] = True
                    fuel_injected = FUEL_FLOW_RATE_LPS * frame_time  # Calculate fuel injected
                    self.total_fuel_injected += fuel_injected
                    print(f"Fueling detected! Fuel injected: {fuel_injected:.2f} L, Total: {self.total_fuel_injected:.2f} L")

        # Fire and Smoke Detection
        fire_detected, smoke_detected = detect_fire_smoke(frame)
        self.alert_status['fire'] = fire_detected
        self.alert_status['smoke'] = smoke_detected

        # Draw Car Bounding Boxes
        for car_box in car_results[0].boxes:
            x1, y1, x2, y2 = map(int, car_box.xyxy[0].cpu().numpy())
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Box for Cars
            cv2.putText(processed_frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Debugging output to check detections
        print(f"Detected {len(car_results[0].boxes)} cars, {len(nozzle_results[0].boxes)} fuel nozzles.")

        # Overlay pose skeleton on detected persons
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
        """Checks if two bounding boxes touch or overlap."""
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

        # Compute time difference for fuel flow calculation
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
    video_path = "C:/Users/pbani/Downloads/istockphoto-2166648349-640_adpp_is.mp4"  # Replace with your video file path
    main(video_path)

