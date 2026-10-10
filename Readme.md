# 🏗️ Smart Helmet Detection System Using ESP32-CAM & YOLOv8

An AI-powered monitoring system that uses an ESP32-CAM to stream live video, and a Python Flask dashboard running YOLOv8 to detect whether a person is wearing a helmet. If the AI detects the specific target, the system instantly sends an alert with a photo directly to a Telegram bot.

## 🛠️ Hardware Requirements
- ESP32-CAM Module
- FTDI Programmer
- Jumper wires & Breadboard
- Powerbank (or stable 5V power supply)

## 💻 Software Requirements
- Arduino IDE (with ESP32 board package installed)
- Python 3.x
- Telegram Bot API
- Pre-trained YOLOv8 Helmet model (`.pt` file)

---

## 🚀 Setup Instructions

### 1. ESP32-CAM Setup (The Camera)
1. Open the standard `CameraWebServer` example in your Arduino IDE.
2. **Configuration to change:** Before uploading, update your Wi-Fi credentials in the code:
   ```cpp
   const char *ssid = "YOUR_WIFI_SSID_HERE";
   const char *password = "YOUR_WIFI_PASSWORD_HERE";
   ```
3. Connect `GPIO 0` to `GND` to put the board in flash mode, and upload the code using your FTDI programmer.
4. **Important:** Remove the `GPIO 0` wire, open the Serial Monitor (115200 baud), and press the RESET button on the ESP32. 
5. Note down the **IP address** printed on the screen (e.g., `http://192.168.1.50`).

### 2. Python Dashboard Setup (The AI Brain)
1. Install the required Python libraries on your computer:
   ```bash
   pip install flask opencv-python ultralytics requests
   ```
2. Open `dashboard.py` in a text editor and update the following configuration variables at the top of the file:
   - **`ESP32_STREAM_URL`**: Change this to the IP address from step 1. Make sure it includes the `:81/stream` port! (Example: `"http://192.168.1.50:81/stream"`)
   - **`TELEGRAM_BOT_TOKEN`**: Paste your Telegram Bot API token here.
   - **`TELEGRAM_CHAT_ID`**: Paste your Telegram Chat ID here.
   - **`MODEL_PATH`**: Ensure this exactly matches the name of your YOLO model file (e.g., `"helmet_yolov8n.pt"`).

### 3. Running the System
1. Plug your flashed ESP32-CAM directly into your powerbank and mount it where you want to monitor.
2. Open a terminal in the project folder and run the dashboard:
   ```bash
   python dashboard.py
   ```
3. Open your web browser and go to `http://127.0.0.1:5000` to view the dark-themed live monitoring dashboard.
4. When you walk into the camera frame with a helmet on, a colored box will track you, and your phone will receive a Telegram alert with a photo!

---

*Note: Alert photos are temporarily saved to the `/alert/` folder to prevent filling up storage space.*
