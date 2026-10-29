# 🎭 Smart Face Mask Detection System for ATMs & Restricted Areas

A real-time IoT security surveillance system designed to detect individuals wearing unauthorized face masks in sensitive areas such as ATMs, bank kiosks, and restricted entrances. When a masked individual is detected, the system immediately sounds a physical buzzer alarm on-site, notifies security via Telegram with a photo, and triggers a browser alarm on the web dashboard.

---

## 🛠️ Tools & Technologies Used

### Hardware
- **ESP32-CAM Module** (AI-Thinker, OV2640 camera) — Video capture and streaming
- **Buzzer** (Passive or Active connected to GPIO 13) — On-site audible deterrence
- **FTDI USB-to-TTL Programmer** — Flashing firmware to ESP32-CAM
- **5V DC Power Supply / Powerbank** — Powering the ESP32-CAM unit
- **Breadboard & Jumper Wires** — Hardware wiring

### Software & Libraries
- **Arduino IDE** (with ESP32 board support) — ESP32 firmware development
- **Python 3.x** — Backend inference and server engine
- **Ultralytics YOLOv8** — Deep learning object detection (`face_mask_detector.pt`)
- **OpenCV (`cv2`)** — Frame capture, image processing, and bounding box rendering
- **Flask** — Web framework serving MJPEG stream and status APIs
- **Web Audio API & HTML5** — Interactive dark-themed CCTV monitoring dashboard
- **Telegram Bot API** — Instant mobile photo notifications to security staff

---

## ⚙️ How It Works

```
[ ESP32-CAM ]
   │
   ├── Port 81: Streams live MJPEG video (/stream) ──────► [ Python Flask Dashboard ]
   │                                                               │
   │                                                      Runs YOLOv8 Detection
   │                                                      (face_mask_detector.pt)
   │                                                               │
   │                                                     Mask Detected? (conf >= 0.65)
   │                                                               │
   └── Port 82: Receives instant /buzzer trigger ◄─────────────────┤
            │                                                      ├──► Telegram Photo Alert
       Sounds Buzzer                                               └──► Web Dashboard Siren
```

1. **Dual-Server ESP32 Architecture**:
   - **Port 81 (`/stream`)**: Serves continuous live video without interruption.
   - **Port 82 (`/buzzer`)**: Dedicated HTTP server handling alarm requests so the buzzer triggers instantly without getting blocked by the video stream.
2. **AI Inference (`dashboard.py`)**:
   - Frames are fetched from `:81/stream` and processed using YOLOv8.
   - A confidence threshold of **0.65** ensures bare faces are not falsely flagged.
3. **Immediate Multi-Tier Alerting**:
   - 🔔 **Physical Alarm**: Dispatches HTTP `GET :82/buzzer` to ring the onboard buzzer for 4 seconds.
   - 📱 **Telegram Alert**: Uploads a timestamped snapshot of the violator to security personnel.
   - 🖥️ **Web Dashboard**: Displays the bounding box live, flashes a warning badge, and sounds a rhythmic siren via the browser's Web Audio API.

---

## 🔌 Hardware Wiring

| Component | ESP32-CAM Pin |
|---|---|
| **Buzzer (+)** | **GPIO 13** |
| **Buzzer (-)** | **GND** |
| **FTDI TX** | **U0R (GPIO 3)** |
| **FTDI RX** | **U0T (GPIO 1)** |
| **FTDI VCC** | **5V** |
| **FTDI GND** | **GND** |
| *Flashing only:* **GPIO 0** | **GND** (remove after upload) |

> 💡 **Buzzer Type**: The sketch is pre-configured for a **passive buzzer** using `tone(BUZZER_PIN, 2700)`. If using an active buzzer, set `const bool IS_ACTIVE_BUZZER = true;` in `ESP32_Stream.ino`.

---

## 🚀 Setup & Execution

### 1. Flash the ESP32-CAM
1. Open [`ESP32_Stream/ESP32_Stream.ino`](ESP32_Stream/ESP32_Stream.ino) in Arduino IDE.
2. Update your WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Select board **AI Thinker ESP32-CAM**.
4. Bridge **GPIO 0 to GND**, plug in FTDI programmer, and click **Upload**.
5. Once uploaded, remove the GPIO 0 jumper, open Serial Monitor at **115200 baud**, and press the **RST** button to get the ESP32 IP address (e.g., `172.20.10.3`).

### 2. Configure & Run Python Backend
1. Install required packages:
   ```bash
   pip install flask opencv-python ultralytics requests
   ```
2. Verify IP and settings in [`dashboard.py`](dashboard.py):
   ```python
   ESP32_STREAM_URL = "http://<YOUR_ESP32_IP>:81/stream"
   ESP32_BUZZER_URL = "http://<YOUR_ESP32_IP>:82/buzzer"
   TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
   TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
   ```
3. Start the application:
   ```bash
   python dashboard.py
   ```
4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
   Click anywhere on the web dashboard to enable audio alarm playback.
