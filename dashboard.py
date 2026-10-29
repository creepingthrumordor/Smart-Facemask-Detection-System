from collections import deque
import threading
import time
import cv2
from flask import Flask, Response, render_template
import requests
from ultralytics import YOLO

app = Flask(__name__)

ESP32_STREAM_URL = "http://172.20.10.3:81/stream"
ESP32_BUZZER_URL = "http://172.20.10.3:82/buzzer"  # Buzzer runs on separate port (82) to avoid stream blocking

TELEGRAM_BOT_TOKEN = "8933831008:AAGQlWVH0VWQBnw52h8tAdaIJOShPyKxmjI"
TELEGRAM_CHAT_ID = "8712314502"

MODEL_PATH = "face_mask_detector.pt"

print(f"Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# --- Print class names at startup for verification ---
print("Model classes:")
for idx, name in model.names.items():
    print(f"  {idx}: '{name}'")


CLASS_ROLE = {}
for idx, name in model.names.items():
    lower_name = name.strip().lower()
    # Check for negative/compliant keywords first
    if any(
        kw in lower_name for kw in ["without", "no-", "no_", "no ", "tidak", "bare"]
    ):
        if any(
            item in lower_name for item in ["helmet", "mask", "hardhat", "bermasker"]
        ):
            CLASS_ROLE[idx] = "compliant"
        else:
            CLASS_ROLE[idx] = "ignored"
    elif any(kw in lower_name for kw in ["mask", "bermasker", "helmet", "hardhat"]):
        CLASS_ROLE[idx] = "violation"
    else:
        CLASS_ROLE[idx] = "ignored"

CONF_THRESHOLD = 0.65  # Raised to 0.65 to eliminate false positives (bare faces misclassified as "mask")
MODEL_IMGSZ = 384  # Optimized: faster inference, still accurate for masks/helmets

# Cooldown for alerts (in seconds)
ALERT_COOLDOWN = 10
last_alert_time = 0

# Temporal smoothing: track per-frame verdicts over a sliding window
verdict_history = deque(maxlen=3)  # Reduced from 5 for faster response

# Alert trigger condition: fire on the very first confident detection
FRAMES_TO_TRIGGER = 1

print("CLASS_ROLE mapping:")
for idx, role in CLASS_ROLE.items():
    print(f"  class {idx} '{model.names[idx]}' -> {role}")


def _send_telegram_worker(frame_copy, label, conf):
    """Worker function to upload photo asynchronously without freezing the stream."""
    temp_img_path = "alert/alert.jpg"
    try:
        cv2.imwrite(temp_img_path, frame_copy)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        detected_item = label if label else "Face covering / Helmet"
        caption = (
            f"⚠️ SECURITY ALERT: Unauthorized {detected_item} detected!\n"
            f"• Confidence: {conf:.1%}\n"
            f"• Model: {MODEL_PATH}\n"
            f"• Status: Restricted entrance access violation"
        )

        with open(temp_img_path, "rb") as img_file:
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
            }
            files = {"photo": img_file}
            response = requests.post(url, data=payload, files=files, timeout=10)
            if response.status_code == 200:
                print("Telegram alert sent successfully!")
            else:
                print(f"Failed to send alert: {response.text}")
    except Exception as e:
        print(f"Error sending telegram alert: {e}")


# Cooldown for buzzer alerts (in seconds)
BUZZER_COOLDOWN = 3  # Reduced from 5 for faster re-triggering
last_buzzer_time = 0
latest_alarm_time = 0


def _trigger_buzzer_worker():
    """Sends HTTP request to ESP32 buzzer server (port 82) to trigger the alarm."""
    try:
        resp = requests.get(ESP32_BUZZER_URL, timeout=2)
        if resp.status_code == 200:
            print("ESP32 Buzzer alarm triggered successfully!")
        else:
            print(f"ESP32 Buzzer returned status code: {resp.status_code}")
    except Exception as e:
        print(f"Buzzer trigger note: Could not reach ESP32 at {ESP32_BUZZER_URL}: {e}")


def trigger_buzzer_alert():
    """Triggers the ESP32 buzzer alarm in a background thread."""
    global last_buzzer_time, latest_alarm_time
    current_time = time.time()
    latest_alarm_time = current_time

    if current_time - last_buzzer_time > BUZZER_COOLDOWN:
        last_buzzer_time = current_time
        print(f"Dispatching ESP32 buzzer alert to {ESP32_BUZZER_URL}...")
        threading.Thread(
            target=_trigger_buzzer_worker, daemon=True
        ).start()


def send_telegram_alert(frame, label, conf):
    """Spawns a thread for sending the alert to avoid blocking video inference."""
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time > ALERT_COOLDOWN:
        last_alert_time = current_time
        print(f"Dispatching Telegram Alert in background ({label} - {conf:.2f})...")
        threading.Thread(
            target=_send_telegram_worker, args=(frame.copy(), label, conf), daemon=True
        ).start()


def generate_frames():
    """Reads from ESP32 stream and executes YOLO inference."""
    cap = cv2.VideoCapture(ESP32_STREAM_URL)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from stream. Retrying in 2 seconds...")
            time.sleep(2)
            cap = cv2.VideoCapture(ESP32_STREAM_URL)
            continue

        results = model(
            frame, stream=True, conf=CONF_THRESHOLD, imgsz=MODEL_IMGSZ, iou=0.45
        )

        # Track detections in this frame
        best_violation_conf = 0.0
        best_violation_label = ""
        violation_count = 0

        best_compliant_conf = 0.0
        best_compliant_label = ""
        compliant_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                role = CLASS_ROLE.get(cls_id, "ignored")

                if role == "violation":
                    violation_count += 1
                    if conf_score > best_violation_conf:
                        best_violation_conf = conf_score
                        best_violation_label = model.names[cls_id]
                elif role == "compliant":
                    compliant_count += 1
                    if conf_score > best_compliant_conf:
                        best_compliant_conf = conf_score
                        best_compliant_label = model.names[cls_id]

            # Draw the YOLO bounding boxes and labels on the frame
            frame = r.plot()

        # --- DECISION LOGIC ---
        frame_verdict = "none"

        if violation_count > 0:
            if compliant_count > 0:
                # If both detected, require violation confidence to be competitive
                if best_violation_conf >= (best_compliant_conf - 0.05):
                    frame_verdict = "violation"
                else:
                    frame_verdict = "compliant"
            else:
                frame_verdict = "violation"
        elif compliant_count > 0:
            frame_verdict = "compliant"

        verdict_history.append(frame_verdict)

        # Count votes in the sliding window
        violation_votes = sum(1 for v in verdict_history if v == "violation")
        compliant_votes = sum(1 for v in verdict_history if v == "compliant")

        # Trigger alert only when violation is consistently detected
        if violation_votes >= FRAMES_TO_TRIGGER and violation_votes > compliant_votes:
            print(f"[TRIGGER] Mask detected! violation_votes={violation_votes} — firing buzzer")
            send_telegram_alert(frame, best_violation_label, best_violation_conf)
            trigger_buzzer_alert()
            verdict_history.clear()  # Flush history to avoid re-alerting immediately


        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/alert_status")
def alert_status():
    """Returns JSON indicating if an alert is actively alarming."""
    is_active = (time.time() - latest_alarm_time) < 2.5
    return {"alarm": is_active}


if __name__ == "__main__":
    print("Starting Monitoring Dashboard...")
    print(f"Active model: {MODEL_PATH}")
    print("Go to http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)
