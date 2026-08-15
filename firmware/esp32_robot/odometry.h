/*
 * odometry.h — quadrature encoder counting.
 *
 * The counters are updated from interrupts and read from the main loop, so
 * they are declared volatile and read inside a critical section. Reading a
 * 32-bit volatile without one is *usually* atomic on the ESP32 and
 * occasionally is not, which produces a rare corrupted count that is very
 * hard to debug later.
 */

#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <Arduino.h>
#include "config.h"

volatile long encoderLeftTicks = 0;
volatile long encoderRightTicks = 0;

portMUX_TYPE encoderMux = portMUX_INITIALIZER_UNLOCKED;

/*
 * Full quadrature decoding: both channels are watched, and the state of the
 * opposite channel at each edge gives the direction.
 *
 * Counting only one channel's rising edge would quarter the resolution and,
 * worse, lose direction entirely — a robot reversing would report that it
 * kept moving forward.
 */
void IRAM_ATTR onLeftEncoderA() {
  bool a = digitalRead(PIN_ENC_LEFT_A);
  bool b = digitalRead(PIN_ENC_LEFT_B);
  portENTER_CRITICAL_ISR(&encoderMux);
  encoderLeftTicks += (a == b) ? 1 : -1;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void IRAM_ATTR onLeftEncoderB() {
  bool a = digitalRead(PIN_ENC_LEFT_A);
  bool b = digitalRead(PIN_ENC_LEFT_B);
  portENTER_CRITICAL_ISR(&encoderMux);
  encoderLeftTicks += (a != b) ? 1 : -1;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void IRAM_ATTR onRightEncoderA() {
  bool a = digitalRead(PIN_ENC_RIGHT_A);
  bool b = digitalRead(PIN_ENC_RIGHT_B);
  portENTER_CRITICAL_ISR(&encoderMux);
  encoderRightTicks += (a == b) ? 1 : -1;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void IRAM_ATTR onRightEncoderB() {
  bool a = digitalRead(PIN_ENC_RIGHT_A);
  bool b = digitalRead(PIN_ENC_RIGHT_B);
  portENTER_CRITICAL_ISR(&encoderMux);
  encoderRightTicks += (a != b) ? 1 : -1;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void setupEncoders() {
  pinMode(PIN_ENC_LEFT_A, INPUT_PULLUP);
  pinMode(PIN_ENC_LEFT_B, INPUT_PULLUP);
  pinMode(PIN_ENC_RIGHT_A, INPUT_PULLUP);
  pinMode(PIN_ENC_RIGHT_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LEFT_A), onLeftEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LEFT_B), onLeftEncoderB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RIGHT_A), onRightEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RIGHT_B), onRightEncoderB, CHANGE);
}

/* Read both counters as one consistent pair. */
void readEncoders(long *left, long *right) {
  portENTER_CRITICAL(&encoderMux);
  *left = encoderLeftTicks;
  *right = encoderRightTicks;
  portEXIT_CRITICAL(&encoderMux);
}

void resetEncoders() {
  portENTER_CRITICAL(&encoderMux);
  encoderLeftTicks = 0;
  encoderRightTicks = 0;
  portEXIT_CRITICAL(&encoderMux);
}

/* Wheel speed in RPM, from a tick delta over a known interval. */
float ticksToRpm(long tickDelta, unsigned long intervalMs) {
  if (intervalMs == 0) return 0.0f;
  float revolutions = (float)tickDelta / (float)TICKS_PER_REVOLUTION;
  return revolutions * 60000.0f / (float)intervalMs;
}

#endif  // ODOMETRY_H
