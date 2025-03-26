from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time
import threading

app = Flask(__name__)
app.config['TEMP_FOLDER'] = 'temp'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Shared processing state
processing_state = {
    'status': 'idle',
    'message': '',
    'alerts': [],
    'fuel': 0.0,
    'progress': 0,
    'logs': [],
    'output_file': None
}

class FuelStationSafetyMonitor:
    def __init__(self, video_path):
        self.models = {
            'nozzle': YOLO("C:/Users/pbani/Downloads/ACEhackathon/runs/detect/fuel_nozzle_detection/weights/best.pt"),
            'car': YOLO("C:/Users/pbani/Downloads/ACEhackathon/Acehackathon/car.pt"),
            'pose': YOLO("C:/Users/pbani/Downloads/ACEhackathon/yolov8s-pose.pt"),
            'fire_smoke': YOLO("C:/Users/pbani/Downloads/ACEhackathon/YOLOv8-Fire-and-Smoke-Detection-main/runs/detect/train/weights/best.pt"),
            'plate': YOLO("C:/Users/pbani/Downloads/ACEhackathon/numberplate.pt")
        }
        
        self.alert_status = {
            'fuel_nozzle': False,
            'fire': False, 
            'smoke': False,
            'pose': False,
            'fueling': False
        }
        self.video_path = video_path
        self.total_fuel_injected = 0.0
        self.FUEL_FLOW_RATE_LPS = 0.095
        self.total_frames = 0
        self.processed_frames = 0

    def process_frame(self, frame, frame_time):
        # Run all detectors
        results = {name: model(frame) for name, model in self.models.items()}
        
        # Start with original frame
        processed_frame = frame.copy()

        # Plot all detections
        for name in ['nozzle', 'car', 'pose', 'fire_smoke', 'plate']:
            processed_frame = results[name][0].plot(img=processed_frame)

        # Fuel nozzle and car collision detection
        nozzle_boxes = [box.xyxy.cpu().numpy() for box in results['nozzle'][0].boxes]
        car_boxes = [box.xyxy.cpu().numpy() for box in results['car'][0].boxes]
        
        self.alert_status['fueling'] = False
        for nozzle_box in nozzle_boxes:
            for car_box in car_boxes:
                if self.boxes_touch(nozzle_box[0], car_box[0]):
                    self.alert_status['fueling'] = True
                    self.total_fuel_injected += self.FUEL_FLOW_RATE_LPS * frame_time

        # Update other alerts
        self.alert_status['fuel_nozzle'] = len(nozzle_boxes) > 0
        self.alert_status['fire'] = any(box.cls.cpu().numpy() == 0 for box in results['fire_smoke'][0].boxes)
        self.alert_status['smoke'] = any(box.cls.cpu().numpy() == 1 for box in results['fire_smoke'][0].boxes)
        self.alert_status['pose'] = results['pose'][0].keypoints.xy.any() if results['pose'] else False

        self.processed_frames += 1
        return processed_frame

    def boxes_touch(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    def save_processed_video(self, output_path):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            raise ValueError("Could not create video writer")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame, 1/fps)
            out.write(processed_frame)
            
            # Update progress
            progress = (self.processed_frames / self.total_frames) * 100
            processing_state.update({
                'fuel': self.total_fuel_injected,
                'alerts': [k for k, v in self.alert_status.items() if v],
                'progress': progress,
                'logs': [f"Processed {self.processed_frames}/{self.total_frames} frames"]
            })
        
        cap.release()
        out.release()

def process_video(filepath):
    try:
        original_filename = os.path.basename(filepath)
        output_filename = f"processed_{original_filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Create directories if needed
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        # Initialize processing state
        processing_state.update({
            'status': 'processing',
            'message': '',
            'alerts': [],
            'fuel': 0.0,
            'progress': 0,
            'logs': [],
            'output_file': None
        })
        
        # Process and save video
        monitor = FuelStationSafetyMonitor(filepath)
        monitor.save_processed_video(output_path)
        
        # Final update
        processing_state.update({
            'status': 'complete',
            'message': 'Video processing completed',
            'output_file': output_filename,
            'fuel': monitor.total_fuel_injected,
            'progress': 100
        })

    except Exception as e:
        processing_state.update({
            'status': 'error',
            'message': str(e)
        })
        # Cleanup failed output
        if os.path.exists(output_path):
            os.remove(output_path)
    finally:
        # Always delete the temporary uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)
    
    thread = threading.Thread(target=process_video, args=(filepath,))
    thread.start()
    
    return jsonify({'message': 'Processing started'})

@app.route('/status')
def get_status():
    return jsonify(processing_state)

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)