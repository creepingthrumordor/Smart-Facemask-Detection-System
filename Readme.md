# Smart Security Surveillance: Real-Time Face Mask Detection for ATMs and Restricted Access

This project explores an IoT-enabled security system for ATM kiosks, banks, and other restricted entrances where face coverings can conceal a person's identity. An ESP32-CAM sends video to a Python edge-processing application, which uses YOLOv8 and MediaPipe to identify configured face-mask classes and hands covering a face. A detected violation can trigger an ESP32 buzzer, a Telegram photo notification, and an alert on the live Flask dashboard.

The project is associated with the **Smart Lanka** thematic track. The abstract lists **Afil Musthafa M** and **Kiruparasa Kavinjan** as authors. Its motivation is to reduce the reliance on continuous manual visual screening with a low-cost wireless camera and automated alerts.

## Project files

| Path | Purpose |
|---|---|
| `dashboard.py` | Flask server, video capture, YOLO inference, MediaPipe face/hand detection, and alert dispatch |
| `templates/index.html` | Browser monitoring dashboard |
| `ESP32_Stream/ESP32_Stream.ino` | ESP32-CAM MJPEG stream and buzzer HTTP server |
| `face_mask_detector.pt` | YOLO model used for object detection |
| `blaze_face_short_range.tflite`, `hand_landmarker.task` | MediaPipe face and hand model assets |

## How it works

1. The ESP32-CAM provides an MJPEG stream at `http://<ESP32_IP>:81/stream`.
2. `dashboard.py` reads the stream and runs YOLO inference (confidence threshold `0.48`, image size `384`). Model classes are assigned a role from their names: mask classes are violations, recognized negative classes such as “without mask” are compliant, and other classes are ignored. If both compliant and violation detections exist, the compliant result suppresses the violation only when its confidence is more than `0.20` higher.
3. MediaPipe face and hand landmarks are also checked for hands overlapping a face. A hand-covering-face result is treated as a violation.
4. On a violation, the backend sends a photo to Telegram (10-second cooldown) and requests the ESP32 buzzer (3-second cooldown). The dashboard polls `/alert_status` and can play a browser-generated alarm. Browser audio may require a user interaction first.

The ESP32 runs two HTTP servers because its continuous stream handler occupies the stream server: port `81` serves `/stream`, and port `82` serves `/buzzer`. The buzzer endpoint activates GPIO 13 for four seconds. The sketch defaults to a passive buzzer; set `IS_ACTIVE_BUZZER` to `true` for an active buzzer. The project abstract describes the design as using a high-resolution alert photo and reports evaluation across indoor lighting conditions and camera angles; those evaluation results are claims from the abstract and are not independently reproduced by the application code or this README.

## Requirements

- AI-Thinker ESP32-CAM with OV2640 camera and a compatible Arduino ESP32 core
- Python and the packages imported by the backend: Flask, OpenCV, MediaPipe, Requests, and Ultralytics
- A working network connection between the ESP32-CAM and the computer running Flask
- Telegram bot credentials and a chat ID if Telegram notifications are required

Install the Python packages with:

```bash
python -m pip install flask opencv-python mediapipe requests ultralytics
```

The two MediaPipe model files are included. If either is missing, the application attempts to download it at startup. The YOLO model file must be present at `face_mask_detector.pt` in the working directory.

## Configure and run

### 1. Flash the ESP32-CAM

1. Open `ESP32_Stream/ESP32_Stream.ino` in Arduino IDE and enter the Wi-Fi SSID and password in the sketch.
2. Select the AI Thinker ESP32-CAM board and the appropriate upload settings for your board.
3. For flashing, connect GPIO 0 to GND, upload the sketch, then remove that connection and reset the board.
4. Open Serial Monitor at 115200 baud. Note the IP address printed after Wi-Fi connects.

Connect a buzzer positive lead to GPIO 13 and its negative lead to GND. Use a suitable driver circuit if the buzzer draws more current than an ESP32 GPIO can provide. For serial flashing, connect the USB-to-TTL adapter TX to U0R, RX to U0T, 5V to 5V, and GND to GND.

### 2. Configure the Python backend

Set the stream URL using the `VIDEO_SOURCE` environment variable, or edit its default in `dashboard.py`:

```bash
VIDEO_SOURCE=http://<ESP32_IP>:81/stream python dashboard.py
```

On Windows PowerShell:

```powershell
$env:VIDEO_SOURCE = "http://<ESP32_IP>:81/stream"
python dashboard.py
```

The buzzer URL and Telegram bot token/chat ID are currently set directly in `dashboard.py`; update them there for your setup. The buzzer endpoint must be `http://<ESP32_IP>:82/buzzer`. Keep bot credentials private and do not commit real credentials to a public repository.

Create the directory used for Telegram alert snapshots before starting the application:

```bash
mkdir -p alert
```

Run from the project directory so the relative model and template paths resolve:

```bash
python dashboard.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The server binds to `0.0.0.0:5000`, so it is also reachable from other devices on the same network using the host computer's IP address.

## Dashboard and routes

- `/` — monitoring dashboard
- `/video_feed` — annotated MJPEG feed produced by the Python backend
- `/alert_status` — JSON status used by the dashboard alarm indicator
- ESP32 `:81/stream` — raw camera MJPEG stream
- ESP32 `:82/buzzer` — HTTP GET endpoint that activates the physical buzzer

## Troubleshooting

- **No camera feed:** confirm the ESP32 IP and that `http://<ESP32_IP>:81/stream` is reachable from the computer running Flask. Check that both devices are on the same network.
- **Buzzer does not sound:** check GPIO 13 wiring, buzzer type, and the `IS_ACTIVE_BUZZER` setting; confirm the ESP32 is reachable on port 82.
- **Telegram photo not sent:** verify the bot token and chat ID in `dashboard.py`, network access, and that the `alert` directory exists and is writable.
- **Model or template not found:** launch `python dashboard.py` from the project directory and ensure `face_mask_detector.pt` and `templates/index.html` are present.
- **Dashboard alarm is silent:** click the page once to allow the browser to start audio, and check the audio toggle.
