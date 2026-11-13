#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"           // Disable brownout problems
#include "soc/rtc_cntl_reg.h"  // Disable brownout problems
#include "esp_http_server.h"

// ===========================
// Enter your WiFi credentials                 
// ===========================
const char* ssid = "john";
const char* password = "00000000";

#define PART_BOUNDARY "123456789000000000000987654321"

// OV2640 camera module pins (AI-THINKER Model)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===========================
// Buzzer Alarm Configuration
// ===========================
// Connect Buzzer (+) to GPIO 13 (Pin 4 on Left Header: 5V, GND, IO12, IO13)
// Connect Buzzer (-) to any GND pin
#define BUZZER_PIN 13
const bool IS_ACTIVE_BUZZER = false; // set to false for passive buzzers (requires AC frequency oscillation)
volatile unsigned long buzzer_until = 0;

httpd_handle_t stream_httpd = NULL;
httpd_handle_t buzzer_httpd = NULL;  // Separate server so buzzer works while streaming

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      if (fb->format != PIXFORMAT_JPEG) {
        // Software compression: quality 40 balances speed and clarity for YOLO detection
        bool jpeg_converted = frame2jpg(fb, 40, &_jpg_buf, &_jpg_buf_len);
        esp_camera_fb_return(fb);
        fb = NULL;
        if (!jpeg_converted) {
          Serial.println("JPEG compression failed");
          res = ESP_FAIL;
        }
      } else {
        _jpg_buf_len = fb->len;
        _jpg_buf = fb->buf;
      }
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

static esp_err_t buzzer_handler(httpd_req_t *req) {
  // Trigger loud alarm for 4 seconds upon violation detection
  buzzer_until = millis() + 4000;
  Serial.println("ALERT! Buzzer alarm triggered by dashboard detection.");
  const char* resp = "BUZZER_TRIGGERED";
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_send(req, resp, strlen(resp));
  return ESP_OK;
}

void startCameraServer() {
  // ---- SERVER 1: Video stream on port 81 ----
  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 32768;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Stream server started on port 81");
  }

  // ---- SERVER 2: Buzzer commands on port 82 ----
  // Separate server because stream_handler's while(true) loop
  // blocks the httpd task, preventing any other request from being served
  httpd_config_t buzzer_config = HTTPD_DEFAULT_CONFIG();
  buzzer_config.server_port = 82;
  buzzer_config.ctrl_port = 32769;  // Must differ from stream server

  httpd_uri_t buzzer_uri = {
    .uri       = "/buzzer",
    .method    = HTTP_GET,
    .handler   = buzzer_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&buzzer_httpd, &buzzer_config) == ESP_OK) {
    httpd_register_uri_handler(buzzer_httpd, &buzzer_uri);
    Serial.println("Buzzer server started on port 82");
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector
  setCpuFrequencyMhz(240);                    // Maximize ESP32 CPU clock to 240MHz for faster software compression
  
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565; // Compatible with all camera modules (avoids error 0x106)
  config.grab_mode = CAMERA_GRAB_LATEST;  // Always drop old frames to ensure instant real-time video
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  // HVGA (480x320) — half the pixels of VGA, good balance of quality and speed (10-15 FPS)
  config.frame_size = FRAMESIZE_HVGA;
  if (psramFound()) {
    Serial.println(">> PSRAM Found! Using HVGA (480x320) with double buffer");
    config.fb_count = 2;
  } else {
    Serial.println(">> Running on internal DRAM with HVGA");
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
  }
  
  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Tune camera sensor registers for sharper image contrast
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_contrast(s, 2);    // +2 enhances image contrast
    s->set_brightness(s, 1);  // +1 prevents dark silhouettes
    s->set_sharpness(s, 2);   // +2 sharpens image details
  }
  
  // Wi-Fi connection
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected!");
  
  Serial.print("Camera Stream Ready! Access it at: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  
  // Start streaming web server
  startCameraServer();

  // Initialize Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.printf("Buzzer initialized on GPIO %d\n", BUZZER_PIN);
}

bool buzzer_is_on = false;

void loop() {
  if (millis() < buzzer_until) {
    if (!buzzer_is_on) {
      buzzer_is_on = true;
      if (IS_ACTIVE_BUZZER) {
        digitalWrite(BUZZER_PIN, HIGH);
      } else {
        tone(BUZZER_PIN, 2700); // 2700 Hz: standard loud resonant frequency for buzzers
      }
    }
  } else {
    if (buzzer_is_on) {
      buzzer_is_on = false;
      if (IS_ACTIVE_BUZZER) {
        digitalWrite(BUZZER_PIN, LOW);
      } else {
        noTone(BUZZER_PIN);
      }
    }
  }
  delay(5);
}
