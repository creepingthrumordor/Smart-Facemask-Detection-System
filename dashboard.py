import os
import urllib.request
from collections import deque
import threading
import time
import cv2
from flask import Flask, Response, render_template
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from ultralytics import YOLO

app = Flask(__name__)

RAW_VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "http://172.20.10.3:81/stream")
try:
    ESP32_STREAM_URL = int(RAW_VIDEO_SOURCE)
except ValueError:
    ESP32_STREAM_URL = RAW_VIDEO_SOURCE

ESP32_BUZZER_URL = "http://172.20.10.3:82/buzzer"  # Buzzer runs on separate port (82) to avoid stream blocking

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

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
        if any(item in lower_name for item in ["mask", "bermasker"]):
            CLASS_ROLE[idx] = "compliant"
        else:
            CLASS_ROLE[idx] = "ignored"
    elif any(kw in lower_name for kw in ["mask", "bermasker"]):
        CLASS_ROLE[idx] = "violation"
    else:
        CLASS_ROLE[idx] = "ignored"

CONF_THRESHOLD = (
    0.48  # Tuned threshold so both masks and face coverings are detected reliably
)
MODEL_IMGSZ = 384  # Inference image size

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

# --- Initialize MediaPipe Face & Hand Landmarker for Hand-on-Face Detection ---
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
HAND_MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(FACE_MODEL_PATH):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        FACE_MODEL_PATH,
    )

if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_MODEL_PATH,
    )

face_detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        min_detection_confidence=0.4,
    )
)
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
]


def detect_hand_covering_face(frame, yolo_face_boxes):
    """
    Detects if someone is covering their face / mouth / nose with hands.
    Uses MediaPipe Face Detector + Hand Landmarker with fallback to YOLO face bounding boxes.
    Renders visual cues (landmarks, warning box) onto the frame.
    Returns: (is_covering, best_conf, best_label)
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        face_res = face_detector.detect(mp_img)
        hand_res = hand_detector.detect(mp_img)
    except Exception as e:
        print(f"Error in face/hand detection: {e}")
        return False, 0.0, ""

    if not hand_res.hand_landmarks:
        return False, 0.0, ""

    face_targets = []
    for f in face_res.detections:
        bb = f.bounding_box
        fx1 = max(0, bb.origin_x)
        fy1 = max(0, bb.origin_y)
        fx2 = min(w, fx1 + bb.width)
        fy2 = min(h, fy1 + bb.height)
        nose = None
        mouth = None
        if len(f.keypoints) >= 4:
            nose = (f.keypoints[2].x * w, f.keypoints[2].y * h)
            mouth = (f.keypoints[3].x * w, f.keypoints[3].y * h)
        face_targets.append({"box": (fx1, fy1, fx2, fy2), "nose": nose, "mouth": mouth})

    for yb in yolo_face_boxes:
        face_targets.append(
            {"box": (yb[0], yb[1], yb[2], yb[3]), "nose": None, "mouth": None}
        )

    hand_boxes = []
    for hand_lms in hand_res.hand_landmarks:
        hx1 = max(0, int(min(lm.x for lm in hand_lms) * w))
        hy1 = max(0, int(min(lm.y for lm in hand_lms) * h))
        hx2 = min(w, int(max(lm.x for lm in hand_lms) * w))
        hy2 = min(h, int(max(lm.y for lm in hand_lms) * h))
        hand_boxes.append((hx1, hy1, hx2, hy2, hand_lms))

    covering_detected = False
    max_conf = 0.0

    for ft in face_targets:
        fx1, fy1, fx2, fy2 = ft["box"]
        face_w = max(1, fx2 - fx1)
        face_h = max(1, fy2 - fy1)
        face_area = face_w * face_h
        lower_fy1 = fy1 + int(face_h * 0.35)

        for hx1, hy1, hx2, hy2, hand_lms in hand_boxes:
            ix1 = max(fx1, hx1)
            iy1 = max(fy1, hy1)
            ix2 = min(fx2, hx2)
            iy2 = min(fy2, hy2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap_ratio = inter_area / face_area

            l_iy1 = max(lower_fy1, hy1)
            l_inter = max(0, ix2 - ix1) * max(0, iy2 - l_iy1)
            lower_overlap_ratio = l_inter / max(1, face_w * (fy2 - lower_fy1))

            lms_in_lower_face = sum(
                1
                for lm in hand_lms
                if fx1 <= lm.x * w <= fx2
                and lower_fy1 <= lm.y * h <= fy2 + int(face_h * 0.1)
            )

            mouth_covered = False
            if ft["mouth"]:
                mx, my = ft["mouth"]
                mouth_covered = hx1 <= mx <= hx2 and hy1 <= my <= hy2

            nose_covered = False
            if ft["nose"]:
                nx, ny = ft["nose"]
                nose_covered = hx1 <= nx <= hx2 and hy1 <= ny <= hy2

            is_covering = (
                overlap_ratio >= 0.15
                or lower_overlap_ratio >= 0.20
                or lms_in_lower_face >= 3
                or mouth_covered
                or nose_covered
            )

            if is_covering:
                covering_detected = True
                conf = min(0.99, max(0.85, 0.75 + overlap_ratio * 0.5))
                if conf > max_conf:
                    max_conf = conf

                # Draw hand landmarks skeleton
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pts[p1_idx], pts[p2_idx], (0, 255, 255), 2)
                for p in pts:
                    cv2.circle(frame, p, 3, (0, 0, 255), -1)

                # Draw prominent red warning box around hand & face overlap
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)

                label = f"Hand Covering Face {max_conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (hx1, max(0, hy1 - lh - 10)),
                    (hx1 + lw + 10, hy1),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (hx1 + 5, hy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    return covering_detected, max_conf, "Hand Covering Face"


def _send_telegram_worker(frame_copy, label, conf):
    """Worker function to upload photo asynchronously without freezing the stream."""
    temp_img_path = "alert/alert.jpg"
    try:
        cv2.imwrite(temp_img_path, frame_copy)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        detected_item = label if label else "Face covering"
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
        threading.Thread(target=_trigger_buzzer_worker, daemon=True).start()


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
    """Reads from video stream and executes YOLO & Hand Covering Face inference."""
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

        yolo_face_boxes = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                role = CLASS_ROLE.get(cls_id, "ignored")

                coords = [int(v) for v in box.xyxy[0].tolist()]
                yolo_face_boxes.append(coords)

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

        # Check for Hand Covering Face violation
        hand_covered, hand_conf, hand_label = detect_hand_covering_face(
            frame, yolo_face_boxes
        )
        if hand_covered:
            violation_count += 1
            if hand_conf > best_violation_conf:
                best_violation_conf = hand_conf
                best_violation_label = hand_label

        # --- DECISION LOGIC ---
        frame_verdict = "none"

        if hand_covered:
            # Covering face with hand is an absolute violation
            frame_verdict = "violation"
        elif violation_count > 0:
            # Mask detected: only suppress if no-mask confidence is overwhelmingly higher (+0.20)
            if compliant_count > 0 and (
                best_compliant_conf > best_violation_conf + 0.20
            ):
                frame_verdict = "compliant"
            else:
                frame_verdict = "violation"
        elif compliant_count > 0:
            frame_verdict = "compliant"

        verdict_history.append(frame_verdict)

        # Count votes in the sliding window
        violation_votes = sum(1 for v in verdict_history if v == "violation")

        # Trigger alert:
        # - Instant trigger on first confident detection (>= 0.50 or hand covering)
        # - Or if recent frames confirm violation
        # - Past compliant frames no longer block genuine violations
        should_trigger = False
        if frame_verdict == "violation":
            if hand_covered or best_violation_conf >= 0.50:
                should_trigger = True
            elif violation_votes >= 1:
                should_trigger = True

        if should_trigger:
            print(
                f"[TRIGGER] Violation detected ({best_violation_label} - {best_violation_conf:.2f}) — firing buzzer & alerts"
            )
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
