/*
 * config.h — every constant you need to change for your robot.
 *
 * Edit this file and nothing else when adapting the firmware to your
 * hardware. The values that must be MEASURED rather than guessed are marked
 * CALIBRATE; see docs/CALIBRATION.md for the procedure.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ── Identity ────────────────────────────────────────────────────────────────
#define ROBOT_ID "MR3W01"

// ── Link ────────────────────────────────────────────────────────────────────
// The firmware can transmit over WiFi/MQTT, Bluetooth LE, or both at once.
// Keeping both enabled is useful during development: BLE works anywhere,
// while MQTT gives you the full stack.
#define ENABLE_WIFI_MQTT 1
#define ENABLE_BLE       1

#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// IP or hostname of the machine running the mapper service.
#define MQTT_HOST "192.168.1.100"
#define MQTT_PORT 1883
#define MQTT_TOPIC_SENSORS "roommapper/" ROBOT_ID "/sensors/raw"
#define MQTT_TOPIC_COMMAND "roommapper/" ROBOT_ID "/command"

// Nordic UART Service UUIDs — the de-facto standard for BLE serial, and what
// most phone terminal apps expect.
#define BLE_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_TX_UUID      "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define BLE_RX_UUID      "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

// ── Robot geometry ──────────────────────────────────────────────────────────
// CALIBRATE: measure the wheel with callipers under load, not from the
// datasheet. A 2 % error here is a 2 % error in every distance reported.
#define WHEEL_DIAMETER_M 0.065f

// CALIBRATE: distance between the two DRIVEN wheel contact patches. The
// caster carries no odometry information and is ignored.
#define WHEEL_BASE_M 0.150f

// CALIBRATE: counts per full revolution of the WHEEL, after the gearbox.
// For a quadrature encoder on the motor shaft this is
//   PPR x 4 (quadrature edges) x gear ratio.
// e.g. an 11 PPR magnetic encoder behind 34:1 gives 11*4*34 = 1496.
//
// Do not use a 20-slot disc encoder. It resolves ~10 mm of travel, and a
// single count of difference between wheels reads as several degrees of
// turn; in simulation that alone produced 0.56 m of position error around a
// small room, versus 0.11 m at 360 counts.
#define TICKS_PER_REVOLUTION 1496

// CALIBRATE: corrects the two wheels never being exactly equal. 1.0 is
// uncorrected. If the robot curves left when commanded straight, lower it.
#define LEFT_WHEEL_TRIM 1.0f

// ── Pin assignments (ESP32 DevKit v1) ───────────────────────────────────────
// Encoders — must be interrupt-capable. On the ESP32 every GPIO is.
#define PIN_ENC_LEFT_A  34
#define PIN_ENC_LEFT_B  35
#define PIN_ENC_RIGHT_A 32
#define PIN_ENC_RIGHT_B 33

// Motor driver (L298N or TB6612FNG).
#define PIN_MOTOR_LEFT_PWM   25
#define PIN_MOTOR_LEFT_DIR1  26
#define PIN_MOTOR_LEFT_DIR2  27
#define PIN_MOTOR_RIGHT_PWM  14
#define PIN_MOTOR_RIGHT_DIR1 12
#define PIN_MOTOR_RIGHT_DIR2 13

// Ultrasonic sensors. One trigger pin is shared by all of them — they must
// therefore be fired in sequence, never simultaneously, or each will hear
// the others' echoes.
#define ULTRASONIC_COUNT 3
#define PIN_ULTRASONIC_TRIG 5
static const int PIN_ULTRASONIC_ECHO[ULTRASONIC_COUNT] = {18, 19, 21};
// Mounting angle of each sensor relative to the robot's forward axis,
// counter-clockwise positive.
static const float ULTRASONIC_ANGLE_DEG[ULTRASONIC_COUNT] = {0.0f, 90.0f, -90.0f};

// I2C for the IMU (MPU6050).
#define PIN_I2C_SDA 22
#define PIN_I2C_SCL 23

// GPS module on UART2.
#define PIN_GPS_RX 16   // ESP32 receives here; wire to the module's TX
#define PIN_GPS_TX 17
#define GPS_BAUD 9600

#define PIN_BUMPER 4
#define PIN_BATTERY_SENSE 36  // ADC1_CH0, via a divider

// ── Timing ──────────────────────────────────────────────────────────────────
#define TELEMETRY_INTERVAL_MS 100   // 10 Hz
#define ULTRASONIC_TIMEOUT_US 25000 // ~4.3 m; beyond this we call it max range
#define ULTRASONIC_SETTLE_MS 8      // between sensors, so echoes do not overlap

// ── Sensing limits ──────────────────────────────────────────────────────────
#define ULTRASONIC_MAX_RANGE_M 4.0f
#define ULTRASONIC_MIN_RANGE_M 0.02f

// Speed of sound at 20 C, metres per microsecond. The pulse travels to the
// target and back, hence the halving where this is used.
#define SOUND_SPEED_M_PER_US 0.000343f

// ── Battery divider ─────────────────────────────────────────────────────────
// Ratio of (R1+R2)/R2 for the divider feeding PIN_BATTERY_SENSE.
#define BATTERY_DIVIDER_RATIO 3.0f
#define BATTERY_FULL_V 8.4f    // 2S Li-ion fully charged
#define BATTERY_EMPTY_V 6.0f

// ── IMU calibration ─────────────────────────────────────────────────────────
// Samples averaged at boot to estimate the gyro's zero-rate bias. This is not
// optional: an uncalibrated MPU6050 drifts around 0.15 deg/s, which is about
// 19 degrees over a two-minute mapping run — enough to shear a rectangular
// room into a shape that no longer measures as a rectangle.
#define GYRO_CALIBRATION_SAMPLES 500
// The robot MUST be stationary while this runs.
#define GYRO_CALIBRATION_DELAY_MS 2

#endif  // CONFIG_H
