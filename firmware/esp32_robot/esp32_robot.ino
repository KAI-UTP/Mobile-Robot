/*
 * esp32_robot.ino — firmware for the 3-wheel room-mapping robot.
 *
 * WHAT THIS DOES
 *   Reads wheel encoders, an IMU, ultrasonic rangefinders and a GPS module,
 *   packs them into one JSON message, and transmits it 10 times a second over
 *   WiFi/MQTT and/or Bluetooth LE. All mapping happens off-board.
 *
 * WHY MAPPING IS OFF-BOARD
 *   The occupancy grid for a modest room is hundreds of kilobytes, which does
 *   not fit comfortably alongside WiFi buffers in the ESP32's ~320 KB of RAM,
 *   and the room-extraction pass is far too slow to run between telemetry
 *   frames. Keeping the firmware to "measure and transmit" also means a bug
 *   in the mapping logic is fixed by restarting a Python process rather than
 *   by reflashing a robot that may be under a desk.
 *
 * BOARD
 *   ESP32 DevKit v1 (or any ESP32 with two free UARTs).
 *   Arduino IDE: Tools > Board > ESP32 Dev Module.
 *
 * LIBRARIES
 *   PubSubClient  (Nick O'Leary)   — MQTT
 *   ArduinoJson   (Benoit Blanchon) — JSON serialisation, v6 or later
 *   The BLE and Wire libraries ship with the ESP32 core.
 *
 * BEFORE FLASHING
 *   Edit config.h. At minimum set WIFI_SSID, WIFI_PASSWORD, MQTT_HOST, and
 *   TICKS_PER_REVOLUTION to match your encoders.
 *
 * ON POWER-UP
 *   The gyro is calibrated for about one second. KEEP THE ROBOT STILL until
 *   the LED stops blinking, or every map it builds afterwards will be skewed.
 */

#include <Arduino.h>
#include <ArduinoJson.h>

#include "config.h"
#include "odometry.h"
#include "sensors.h"

#if ENABLE_WIFI_MQTT
  #include <WiFi.h>
  #include <PubSubClient.h>
  WiFiClient wifiClient;
  PubSubClient mqttClient(wifiClient);
#endif

#if ENABLE_BLE
  #include <BLEDevice.h>
  #include <BLEServer.h>
  #include <BLEUtils.h>
  #include <BLE2902.h>
  BLECharacteristic *bleTxCharacteristic = nullptr;
  bool bleClientConnected = false;
#endif

// ── State ───────────────────────────────────────────────────────────────────

unsigned long sequence = 0;
unsigned long lastTelemetryMs = 0;
long previousLeftTicks = 0;
long previousRightTicks = 0;

#define LED_PIN 2

// ── BLE plumbing ────────────────────────────────────────────────────────────

#if ENABLE_BLE
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override { bleClientConnected = true; }
  void onDisconnect(BLEServer *server) override {
    bleClientConnected = false;
    // Without this the ESP32 stops advertising after the first disconnect and
    // the phone can never reconnect.
    server->startAdvertising();
  }
};

void setupBLE() {
  BLEDevice::init(ROBOT_ID);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(BLE_SERVICE_UUID);
  bleTxCharacteristic = service->createCharacteristic(
      BLE_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  bleTxCharacteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(BLE_SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.println("[BLE] advertising as " ROBOT_ID);
}

/*
 * Send a payload over BLE, split across notifications.
 *
 * A BLE notification carries only (MTU - 3) bytes, and the default MTU is 23
 * — so 20 bytes per packet. A telemetry frame is several hundred bytes and
 * must be chunked; sending it in one call silently truncates it.
 */
void bleSend(const char *payload, size_t length) {
  if (!bleClientConnected || bleTxCharacteristic == nullptr) return;

  const size_t chunk = 20;
  for (size_t offset = 0; offset < length; offset += chunk) {
    size_t size = min(chunk, length - offset);
    bleTxCharacteristic->setValue((uint8_t *)(payload + offset), size);
    bleTxCharacteristic->notify();
    delay(4);  // let the stack drain; without it notifications are dropped
  }
  // Newline terminator so the receiver knows where the frame ends.
  bleTxCharacteristic->setValue((uint8_t *)"\n", 1);
  bleTxCharacteristic->notify();
}
#endif  // ENABLE_BLE

// ── WiFi and MQTT ───────────────────────────────────────────────────────────

#if ENABLE_WIFI_MQTT
void setupWiFi() {
  Serial.print("[WiFi] connecting");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  // Bounded wait: the robot must still work over BLE with no WiFi present.
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(" ok, ip=");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(" FAILED (continuing without WiFi)");
  }
}

void ensureMqttConnected() {
  if (WiFi.status() != WL_CONNECTED || mqttClient.connected()) return;

  // One non-blocking attempt per call. Looping here would stall telemetry.
  if (mqttClient.connect(ROBOT_ID)) {
    Serial.println("[MQTT] connected");
    mqttClient.subscribe(MQTT_TOPIC_COMMAND);
  }
}

void onMqttMessage(char *topic, byte *payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, payload, length)) return;

  const char *command = doc["command"];
  if (command == nullptr) return;

  if (strcmp(command, "RESET_ODOMETRY") == 0) {
    resetEncoders();
    previousLeftTicks = 0;
    previousRightTicks = 0;
    Serial.println("[CMD] odometry reset");
  } else if (strcmp(command, "CALIBRATE_GYRO") == 0) {
    Serial.println("[CMD] recalibrating gyro — keep the robot still");
    calibrateGyro();
  }
}
#endif  // ENABLE_WIFI_MQTT

// ── Telemetry ───────────────────────────────────────────────────────────────

/*
 * Build one sensor packet.
 *
 * The field names and structure must match SensorPacket in
 * shared/robotmap_common/models.py exactly — that Pydantic model validates
 * every message, and a mismatch shows up as a rejected packet rather than a
 * crash, so it is easy to miss.
 */
size_t buildTelemetryJson(char *buffer, size_t capacity, unsigned long intervalMs) {
  StaticJsonDocument<1024> doc;

  doc["schema_version"] = "1.0";
  doc["robot_id"] = ROBOT_ID;

  // No real-time clock on board. The mapper stamps arrival time; this field
  // exists to satisfy the schema and to expose uptime for debugging.
  char timestamp[32];
  unsigned long seconds = millis() / 1000;
  snprintf(timestamp, sizeof(timestamp), "1970-01-01T%02lu:%02lu:%02luZ",
           (seconds / 3600) % 24, (seconds / 60) % 60, seconds % 60);
  doc["timestamp"] = timestamp;
  doc["sequence"] = sequence++;
  doc["link"] = "WIFI_MQTT";

  long leftTicks, rightTicks;
  readEncoders(&leftTicks, &rightTicks);

  JsonObject encoders = doc.createNestedObject("encoders");
  encoders["left_ticks"] = leftTicks;
  encoders["right_ticks"] = rightTicks;
  encoders["left_rpm"] = ticksToRpm(leftTicks - previousLeftTicks, intervalMs);
  encoders["right_rpm"] = ticksToRpm(rightTicks - previousRightTicks, intervalMs);
  encoders["dt_ms"] = intervalMs;
  previousLeftTicks = leftTicks;
  previousRightTicks = rightTicks;

  float gyroZ;
  updateIMU(&gyroZ);
  float gz, ax, ay, temperature;
  readIMURaw(&gz, &ax, &ay, &temperature);

  JsonObject imu = doc.createNestedObject("imu");
  imu["heading_deg"] = imuHeadingDeg;
  imu["gyro_z_dps"] = gyroZ;
  imu["accel_x_ms2"] = ax;
  imu["accel_y_ms2"] = ay;
  imu["temperature_c"] = temperature;
  imu["calibrated"] = imuCalibrated;

  float distances[ULTRASONIC_COUNT];
  readAllUltrasonic(distances);

  JsonArray ranges = doc.createNestedArray("ranges");
  for (int i = 0; i < ULTRASONIC_COUNT; i++) {
    JsonObject reading = ranges.createNestedObject();
    reading["angle_deg"] = ULTRASONIC_ANGLE_DEG[i];
    // A negative distance means the reading was below the sensor's minimum,
    // which is not a measurement. Flag it invalid so the mapper skips it
    // rather than recording a wall against the robot's nose.
    reading["distance_m"] = distances[i] < 0 ? 0.0f : distances[i];
    reading["valid"] = distances[i] >= 0;
    char id[8];
    snprintf(id, sizeof(id), "s%d", (int)ULTRASONIC_ANGLE_DEG[i]);
    reading["sensor_id"] = id;
  }

  // Only send GPS when the receiver reports a fix. Sending an all-zero
  // position would be indistinguishable from a genuine fix at the equator.
  if (currentFix.valid) {
    JsonObject gps = doc.createNestedObject("gps");
    gps["latitude"] = currentFix.latitude;
    gps["longitude"] = currentFix.longitude;
    gps["altitude_m"] = currentFix.altitude;
    const char *quality = "NO_FIX";
    switch (currentFix.fixQuality) {
      case 1: quality = "GPS"; break;
      case 2: quality = "DGPS"; break;
      case 4: quality = "RTK_FIXED"; break;
      case 5: quality = "RTK_FLOAT"; break;
    }
    gps["fix_quality"] = quality;
    gps["satellites"] = currentFix.satellites;
    gps["hdop"] = currentFix.hdop;
  }

  float voltage = readBatteryVoltage();
  JsonObject power = doc.createNestedObject("power");
  power["battery_v"] = voltage;
  power["battery_soc"] = batteryPercent(voltage);
  power["current_a"] = 0.0f;

  doc["bumper_active"] = digitalRead(PIN_BUMPER) == LOW;

  return serializeJson(doc, buffer, capacity);
}

// ── Setup and loop ──────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("=== Room Mapper robot " ROBOT_ID " ===");

  pinMode(LED_PIN, OUTPUT);
  pinMode(PIN_BUMPER, INPUT_PULLUP);
  analogReadResolution(12);

  setupEncoders();
  setupUltrasonic();
  setupGPS();

  if (setupIMU()) {
    Serial.println("[IMU] found. Calibrating — KEEP THE ROBOT STILL.");
    // Blink through calibration so it is obvious the robot must not be moved.
    for (int i = 0; i < 5; i++) {
      digitalWrite(LED_PIN, HIGH); delay(80);
      digitalWrite(LED_PIN, LOW);  delay(80);
    }
    calibrateGyro();
    Serial.print("[IMU] gyro bias = ");
    Serial.print(gyroBiasZ, 4);
    Serial.println(" deg/s");
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println("[IMU] NOT FOUND — check I2C wiring. Heading will drift badly.");
  }

#if ENABLE_WIFI_MQTT
  setupWiFi();
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);
  mqttClient.setBufferSize(1024);  // the default 256 truncates our packets
#endif

#if ENABLE_BLE
  setupBLE();
#endif

  Serial.println("[READY] streaming telemetry");
  lastTelemetryMs = millis();
}

void loop() {
  updateGPS();  // drain the UART continuously so sentences are not lost

#if ENABLE_WIFI_MQTT
  ensureMqttConnected();
  mqttClient.loop();
#endif

  unsigned long now = millis();
  // Unsigned arithmetic, so this survives the millis() rollover at ~49 days.
  if (now - lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;

  unsigned long interval = now - lastTelemetryMs;
  lastTelemetryMs = now;

  static char payload[1024];
  size_t length = buildTelemetryJson(payload, sizeof(payload), interval);

#if ENABLE_WIFI_MQTT
  if (mqttClient.connected()) {
    mqttClient.publish(MQTT_TOPIC_SENSORS, (const uint8_t *)payload, length, false);
  }
#endif

#if ENABLE_BLE
  bleSend(payload, length);
#endif

  // Heartbeat: a steady blink means packets are going out.
  digitalWrite(LED_PIN, (sequence % 10 < 5) ? HIGH : LOW);
}
