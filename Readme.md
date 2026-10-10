# smart-helmet-detection

A lightweight monitoring system that uses an ESP32-CAM to stream live video, and a Python Flask dashboard running YOLOv8 to detect whether a person is wearing a helmet. If the AI detects the specific target, the system instantly sends an alert with a photo to a Telegram bot.

## Requirements

Hardware:
- ESP32-CAM module
- FTDI programmer
- Jumper wires and breadboard
- Powerbank or 5V power supply

Software:
- Arduino IDE with ESP32 board package
- Python 3.x
- Pre-trained YOLOv8 model (`helmet_yolov8n.pt`)

Python packages:
- `flask`
- `opencv-python`
- `ultralytics`
- `requests`

## Setup

### ESP32-CAM

1. Open the standard `CameraWebServer` example in your Arduino IDE.
2. Update the Wi-Fi credentials in the sketch:
   ```cpp
   const char *ssid = "YOUR_WIFI_SSID_HERE";
   const char *password = "YOUR_WIFI_PASSWORD_HERE";
   ```
3. Connect `GPIO 0` to `GND` and upload the code using your FTDI programmer.
4. Remove the `GPIO 0` wire, open the Serial Monitor (115200 baud), and press RESET. Note the IP address printed.

### Dashboard

1. Install the required Python dependencies:
   ```sh
   pip install flask opencv-python ultralytics requests
   ```
2. Place your `helmet_yolov8n.pt` model in the same directory as the script.
3. Open `dashboard.py` and update the configuration variables at the top of the file:
   - `ESP32_STREAM_URL`: Set to the IP address from the ESP32 (e.g., `"http://192.168.1.50:81/stream"`).
   - `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API token.
   - `TELEGRAM_CHAT_ID`: Your Telegram Chat ID.
   - `MODEL_PATH`: The filename of your model (e.g., `"helmet_yolov8n.pt"`).

## Usage

1. Plug the flashed ESP32-CAM into a powerbank and mount it.
2. Start the monitoring dashboard:
   ```sh
   python dashboard.py
   ```
3. Navigate to `http://127.0.0.1:5000` in your web browser.

Alert photos are temporarily saved to the local `alert` directory and immediately forwarded to Telegram upon detection.
