import cv2
import requests
import time
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)


ESP32_STREAM_URL = "http://172.20.10.2:81/stream" 

TELEGRAM_BOT_TOKEN = "8933831008:AAGQlWVH0VWQBnw52h8tAdaIJOShPyKxmjI"
TELEGRAM_CHAT_ID = "8712314502"

# Absolute path to your trained YOLOv8 model
MODEL_PATH = "helmet_yolov8n.pt"
model = YOLO(MODEL_PATH)

# Cooldown for alerts (in seconds) to prevent spamming Telegram
ALERT_COOLDOWN = 30 
last_alert_time = 0

def send_telegram_alert(frame):
    """Sends a frame with detections to the Telegram bot."""
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > ALERT_COOLDOWN:
        print("Sending Telegram Alert...")
        # Save temporary image for the alert
        temp_img_path = "alert/alert.jpg"
        cv2.imwrite(temp_img_path, frame)
        
        # Send to Telegram via Bot API
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        try:
            with open(temp_img_path, "rb") as img_file:
                payload = {
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "caption": "✅ Helmet detected in the camera frame!"
                }
                files = {"photo": img_file}
                response = requests.post(url, data=payload, files=files)
                if response.status_code == 200:
                    print("Alert sent successfully!")
                    last_alert_time = current_time
                else:
                    print(f"Failed to send alert: {response.text}")
        except Exception as e:
            print(f"Error sending telegram alert: {e}")

def generate_frames():
    """Generator function that reads from ESP32 stream and runs YOLO inference."""
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from stream. Retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            continue
            
        # Run YOLO inference
        results = model(frame, stream=True, conf=0.2)
        
        alert_triggered = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class ID and name
                cls = int(box.cls[0])
                class_name = model.names[cls].lower()
                
                # Check if it detects a helmet
                if class_name in ["with helmet"]:
                    alert_triggered = True

            # Draw bounding boxes on the frame
            frame = r.plot()
            
        if alert_triggered:
            send_telegram_alert(frame)

        # Encode the frame in JPEG format for the web dashboard
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in the byte format for HTTP multipart streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Monitoring Dashboard...")
    print("Go to http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=False)
