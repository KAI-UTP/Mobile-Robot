/*
 * sensors.h — ultrasonic ranging, IMU, GPS and battery.
 */

#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include <Wire.h>
#include "config.h"

// ── Ultrasonic ──────────────────────────────────────────────────────────────

void setupUltrasonic() {
  pinMode(PIN_ULTRASONIC_TRIG, OUTPUT);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  for (int i = 0; i < ULTRASONIC_COUNT; i++) {
    pinMode(PIN_ULTRASONIC_ECHO[i], INPUT);
  }
}

/*
 * Fire one sensor and return the distance in metres.
 *
 * Returns ULTRASONIC_MAX_RANGE_M on timeout. That is a deliberate choice and
 * the mapper depends on it: "no echo" means nothing was within range, not
 * that a wall sits exactly at the range limit. The mapper marks such a ray as
 * free space rather than placing a wall at its end.
 *
 * Returns -1.0 for a reading below the sensor's minimum, which is not a real
 * measurement — below about 2 cm the echo returns before the receiver has
 * stopped ringing from the transmit burst.
 */
float readUltrasonic(int index) {
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  delayMicroseconds(4);
  digitalWrite(PIN_ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);

  unsigned long duration =
      pulseIn(PIN_ULTRASONIC_ECHO[index], HIGH, ULTRASONIC_TIMEOUT_US);

  if (duration == 0) return ULTRASONIC_MAX_RANGE_M;

  // Halved because the pulse makes a round trip.
  float distance = (duration * SOUND_SPEED_M_PER_US) / 2.0f;

  if (distance < ULTRASONIC_MIN_RANGE_M) return -1.0f;
  if (distance > ULTRASONIC_MAX_RANGE_M) return ULTRASONIC_MAX_RANGE_M;
  return distance;
}

/*
 * Read every sensor in sequence.
 *
 * They share a trigger line and, more importantly, share the air. Firing them
 * together means each one hears the others' echoes and reports whichever
 * arrives first — usually a wall that is not in front of it at all. The
 * settle delay lets the previous burst decay before the next is sent.
 */
void readAllUltrasonic(float *distances) {
  for (int i = 0; i < ULTRASONIC_COUNT; i++) {
    distances[i] = readUltrasonic(i);
    delay(ULTRASONIC_SETTLE_MS);
  }
}

// ── MPU6050 IMU ─────────────────────────────────────────────────────────────

#define MPU6050_ADDR 0x68
#define MPU6050_PWR_MGMT_1 0x6B
#define MPU6050_ACCEL_XOUT_H 0x3B
#define MPU6050_GYRO_CONFIG 0x1B

float gyroBiasZ = 0.0f;      // deg/s, measured at boot
float imuHeadingDeg = 0.0f;  // integrated yaw
bool imuCalibrated = false;
unsigned long lastImuMicros = 0;

// Sensitivity for the +/-250 deg/s range: LSB per deg/s.
#define GYRO_SCALE 131.0f

bool mpuWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool setupIMU() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);

  // Wake the device: it boots into sleep mode.
  if (!mpuWrite(MPU6050_PWR_MGMT_1, 0x00)) return false;
  delay(100);
  mpuWrite(MPU6050_GYRO_CONFIG, 0x00);  // +/-250 deg/s, the most sensitive range
  delay(50);
  return true;
}

bool readIMURaw(float *gyroZ, float *accelX, float *accelY, float *tempC) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom(MPU6050_ADDR, 14, true) != 14) return false;

  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();              // az, unused for planar motion
  int16_t rawTemp = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();              // gx, unused
  Wire.read(); Wire.read();              // gy, unused
  int16_t gz = (Wire.read() << 8) | Wire.read();

  *accelX = ax / 16384.0f * 9.81f;       // +/-2 g range
  *accelY = ay / 16384.0f * 9.81f;
  *tempC = rawTemp / 340.0f + 36.53f;    // per the datasheet
  *gyroZ = gz / GYRO_SCALE;
  return true;
}

/*
 * Estimate and store the gyro's zero-rate offset.
 *
 * THE ROBOT MUST BE COMPLETELY STILL while this runs. Every MPU6050 reports a
 * small non-zero rotation rate when stationary; integrated over a mapping
 * run that constant becomes tens of degrees of heading error, which shears
 * the finished map. Averaging a few hundred stationary samples and
 * subtracting the mean removes almost all of it.
 */
void calibrateGyro() {
  double sum = 0.0;
  int valid = 0;

  for (int i = 0; i < GYRO_CALIBRATION_SAMPLES; i++) {
    float gz, ax, ay, t;
    if (readIMURaw(&gz, &ax, &ay, &t)) {
      sum += gz;
      valid++;
    }
    delay(GYRO_CALIBRATION_DELAY_MS);
  }

  if (valid > GYRO_CALIBRATION_SAMPLES / 2) {
    gyroBiasZ = (float)(sum / valid);
    imuCalibrated = true;
  }
  imuHeadingDeg = 0.0f;
  lastImuMicros = micros();
}

/* Integrate the bias-corrected rate into a heading. */
void updateIMU(float *gyroZOut) {
  float gz, ax, ay, t;
  if (!readIMURaw(&gz, &ax, &ay, &t)) {
    *gyroZOut = 0.0f;
    return;
  }

  float corrected = gz - gyroBiasZ;
  unsigned long now = micros();
  // Unsigned subtraction, so this stays correct across the ~71 minute
  // micros() rollover.
  float dt = (now - lastImuMicros) / 1000000.0f;
  lastImuMicros = now;

  if (dt > 0.0f && dt < 1.0f) {
    imuHeadingDeg += corrected * dt;
    while (imuHeadingDeg < 0.0f) imuHeadingDeg += 360.0f;
    while (imuHeadingDeg >= 360.0f) imuHeadingDeg -= 360.0f;
  }
  *gyroZOut = corrected;
}

// ── GPS (NMEA over UART2) ───────────────────────────────────────────────────

struct GpsFix {
  double latitude = 0.0;
  double longitude = 0.0;
  float altitude = 0.0f;
  int fixQuality = 0;
  int satellites = 0;
  float hdop = 99.9f;
  bool valid = false;
};

GpsFix currentFix;
static char nmeaBuffer[100];
static int nmeaIndex = 0;

/* Convert NMEA ddmm.mmmm to decimal degrees. */
double nmeaToDecimal(const char *value, char hemisphere) {
  double raw = atof(value);
  int degrees = (int)(raw / 100);
  double minutes = raw - (degrees * 100);
  double result = degrees + minutes / 60.0;
  if (hemisphere == 'S' || hemisphere == 'W') result = -result;
  return result;
}

/*
 * Parse a GGA sentence, which carries fix quality, satellite count and HDOP.
 *
 * Those three fields are what let the mapper decide whether to believe the
 * position at all. A receiver indoors still emits a latitude and longitude;
 * only the quality fields reveal that it is meaningless. Never transmit the
 * coordinates without them.
 */
bool parseGGA(char *sentence) {
  char *fields[16] = {0};
  int count = 0;
  char *token = strtok(sentence, ",");
  while (token != NULL && count < 16) {
    fields[count++] = token;
    token = strtok(NULL, ",");
  }
  if (count < 9) return false;

  int quality = atoi(fields[6]);
  if (quality == 0) {
    currentFix.valid = false;
    currentFix.fixQuality = 0;
    currentFix.satellites = atoi(fields[7]);
    return true;
  }

  currentFix.latitude = nmeaToDecimal(fields[2], fields[3][0]);
  currentFix.longitude = nmeaToDecimal(fields[4], fields[5][0]);
  currentFix.fixQuality = quality;
  currentFix.satellites = atoi(fields[7]);
  currentFix.hdop = atof(fields[8]);
  currentFix.altitude = (count > 9) ? atof(fields[9]) : 0.0f;
  currentFix.valid = true;
  return true;
}

void setupGPS() {
  Serial2.begin(GPS_BAUD, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);
}

/* Drain the UART. Non-blocking, so it is safe to call every loop. */
void updateGPS() {
  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n' || c == '\r') {
      if (nmeaIndex > 6) {
        nmeaBuffer[nmeaIndex] = '\0';
        if (strstr(nmeaBuffer, "GGA") != NULL) {
          parseGGA(nmeaBuffer);
        }
      }
      nmeaIndex = 0;
    } else if (nmeaIndex < (int)sizeof(nmeaBuffer) - 1) {
      nmeaBuffer[nmeaIndex++] = c;
    } else {
      nmeaIndex = 0;  // overlong sentence: discard rather than overflow
    }
  }
}

// ── Battery ─────────────────────────────────────────────────────────────────

float readBatteryVoltage() {
  // Average a few samples: the ESP32 ADC is noisy, especially with motors running.
  long total = 0;
  for (int i = 0; i < 8; i++) total += analogRead(PIN_BATTERY_SENSE);
  float counts = total / 8.0f;
  float pinVoltage = counts * 3.3f / 4095.0f;
  return pinVoltage * BATTERY_DIVIDER_RATIO;
}

float batteryPercent(float voltage) {
  float pct = (voltage - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V) * 100.0f;
  if (pct < 0.0f) return 0.0f;
  if (pct > 100.0f) return 100.0f;
  return pct;
}

#endif  // SENSORS_H
