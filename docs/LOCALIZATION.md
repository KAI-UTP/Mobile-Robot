# Where GPS and Bluetooth actually fit

You asked to use GPS and Bluetooth to find the robot's position and draw the
room. This document explains what each one can and cannot do here, with
numbers measured from this project's own simulator rather than assertions.

**Short version:** Bluetooth is the data link and works well for that. GPS
georeferences the map — it tells you *which room in which building* — but it
cannot draw the room, and letting it try makes the result worse. The drawing
comes from wheel odometry, an IMU, and range sensors.

---

## 1. Why GPS cannot draw a room

### The error is bigger than the thing being measured

Uncorrected consumer GNSS has a horizontal accuracy of roughly **4 m** in the
open. A typical room is **4–6 m across**. The measurement uncertainty is the
same size as the object.

Indoors it is far worse. The signal has to pass through a roof, and what
reaches the receiver is mostly reflections. You do not simply lose the fix —
you get a *plausible-looking* fix that is wrong by tens of metres and wanders
while the robot sits still. That is the dangerous failure mode: a receiver
indoors will happily report a latitude and longitude that looks entirely
reasonable.

| Environment | Satellites | HDOP | Realistic accuracy |
|---|---|---|---|
| Open field | 8–12 | 0.7–1.5 | 3–5 m |
| Near buildings | 5–8 | 1.5–3 | 5–15 m |
| Indoors, near a window | 0–4 | 6–25 | 20–50 m, or no fix |
| Indoors, interior room | 0 | — | no fix |

### Measured: what happens if you trust it anyway

The simulator models a receiver that degrades indoors exactly as a real one
does. I ran the full mapping pipeline on a **6.0 × 4.5 m room (27.0 m²)**
five times with different noise seeds, once accepting GPS position fixes and
once rejecting them.

| Seed | Area with GPS corrections | Area with GPS rejected |
|---|---|---|
| 1 | 26.4 m² | 26.8 m² |
| 7 | **42.0 m²** | 25.8 m² |
| 42 | **43.5 m²** | 27.0 m² |
| 123 | **51.8 m²** | 27.2 m² |
| 2024 | **53.6 m²** | 26.3 m² |

With GPS steering the pose, the answer ranged from 26 to 54 m² depending on
nothing but the noise draw, and the robot usually failed to recognise it had
completed a lap. With GPS rejected, every run landed within 5 % of truth.

Critically, an outdoor run with GPS *disabled* produced **exactly** the same
result as the indoor run — 25.93 m² in both cases. That rules out the
difference being a coincidence of noise: the degradation is caused by the GPS
corrections themselves.

This is why `FilterConfig.gps_max_accuracy_for_correction_m` exists, and why
it defaults to 1.0 m. A fix may only move the robot on the map if it is more
precise than the map needs. Consumer GPS never is.

---

## 2. What GPS *is* genuinely good for

Rejecting it for positioning does not make it useless. It does three things
nothing else in the system can:

**Georeferencing.** The map's origin is wherever the robot was switched on —
an arbitrary point. Once a sustained run of good fixes arrives, the filter
records the latitude and longitude of that origin. The finished floor plan can
then be placed on a real map, and two rooms surveyed on different days can be
related to each other. This is `anchor_latitude` / `anchor_longitude` in the
pose message.

**Outdoor operation.** You said the robot will work indoors *and* outdoors. In
a car park or field there are no walls to follow and no range returns to map
against, and GPS becomes the only absolute reference available. The accuracy
gate is a threshold, not a ban: raise
`gps_max_accuracy_for_correction_m` when you are mapping something large
enough that 4 m of error is acceptable.

**Detecting that dead reckoning has failed.** Odometry drifts silently. A GPS
fix that disagrees with the estimate by 50 m is proof something has gone
wrong — a wheel slipping continuously, a robot picked up and moved — even
when the fix is far too coarse to correct the error.

### If you genuinely need GPS-grade indoor position

There is a version of your original idea that works, and it is worth knowing
about:

- **RTK GNSS** reaches 1–2 cm and *is* precise enough to map a room — outdoors
  only. A base station plus rover costs roughly RM 1,200–2,500. The code
  already supports it: an `RTK_FIXED` fix has an estimated accuracy of about
  0.02 m, passes the gate, and is used for corrections.
- **Bluetooth AoA beacons** (Bluetooth 5.1 direction finding) reach 10–50 cm
  indoors, but need several fixed anchors surveyed into the room first — which
  rather defeats the purpose of a robot that measures the room for you.
- **UWB** (DW1000 tags) reach 10–30 cm indoors for about RM 400 for four
  anchors. This is the honest answer to "radio positioning indoors".

None of these are necessary for your project. Odometry plus an IMU plus range
sensors already measures the room to within a few percent, which the test
suite demonstrates.

---

## 3. What Bluetooth is for

Bluetooth's job here is the **data link**, and it is a good one.

| Use | Verdict |
|---|---|
| Streaming telemetry robot → laptop | **Yes.** This is what it is for |
| Sending commands laptop → robot | **Yes** |
| Live monitoring from a phone | **Yes**, via BLE |
| Measuring the robot's position by signal strength | **No** — see below |

### Why not RSSI positioning

It is tempting: signal strength falls with distance, so strength should imply
distance. In practice indoor RSSI trilateration lands at **±2–4 m**, no better
than the GPS you just rejected, because received power depends on body
blocking, antenna orientation, multipath, and reflections off metal furniture
far more than on distance. It is also not stable over time — the same robot in
the same spot reports different strengths as people move around the room.

The firmware supports two Bluetooth paths:

- **BLE (ESP32).** Notifications over the Nordic UART Service. Note that a
  BLE notification carries only 20 bytes by default, so telemetry frames are
  chunked — see `bleSend()` in the firmware.
- **Classic SPP (HC-05).** Appears as a serial port on the laptop.
  `services/bt-bridge` reads it and republishes to MQTT, so a robot with no
  WiFi produces an identical topic stream to an ESP32 with WiFi.

---

## 4. What actually draws the room

Three sensors, each covering the others' weaknesses.

### Wheel odometry — the backbone

Counting wheel rotations gives excellent *short-term* position. Its weakness
is that error accumulates without bound and the robot cannot detect it: a
slipping wheel and a turning wheel are the same signal.

Uncertainty is modelled as a random walk, so variance grows linearly with
distance and one-sigma error grows with its square root. At the default
3 cm/m, 100 m of driving predicts about 30 cm of drift.

> A subtle bug worth noting, because it is easy to write and hard to see: an
> earlier version accumulated `(sigma_per_metre × distance)²` per step, which
> made the reported uncertainty depend on the *telemetry rate*. The same metre
> driven reported a hundred times less uncertainty when split across a hundred
> packets. `test_uncertainty_is_independent_of_sampling_rate` now pins this.

### IMU — fixes the heading

Heading error is what destroys a map. An error of one degree at the start of a
6 m wall puts the far end 10 cm out of place, and errors compound around a
circuit.

The IMU's own zero is arbitrary, so the first reading establishes an offset
into the map frame rather than being adopted as the heading. Thereafter the
two are blended by a scalar Kalman gain. The IMU is ignored while the robot is
rotating faster than 200 °/s, where its accelerometer reference is useless.

### Range sensors — find the walls

Odometry says where the robot is; the rangefinders say where the walls are
relative to it. Each reading is projected into world coordinates and written
into a log-odds occupancy grid.

---

## 5. How the pieces combine

```
encoders ─┐
          ├─► pose filter ─► pose + uncertainty ─┐
IMU ──────┤        ▲                             │
          │        │ (gated)                     ▼
GPS ──────┴────────┘                    occupancy grid
                                                 │
                                                 ▼
                                     flood fill → open → fill holes
                                                 │
                                                 ▼
                                    trace → simplify → square up
                                                 │
                                                 ▼
                                     room outline + area in m²
```

The gate on the GPS arrow is the whole point of this document.

---

## 6. Measured performance

From `tests/test_integration.py`, mapping a 6.0 × 4.5 m room by driving one
wall-following lap:

| Quantity | Truth | Measured | Error |
|---|---|---|---|
| Floor area | 27.0 m² | 26.77 m² | 0.9 % |
| Long dimension | 6.00 m | 5.95 m | 0.8 % |
| Short dimension | 4.50 m | 4.50 m | 0.0 % |
| Position drift after one lap | — | 0.11 m | — |
| GPS fixes accepted (indoors) | — | 0 of 1353 | — |

Room dimensions are measured against the **room's own axes** using a
minimum-area enclosing rectangle, not the map's axes. The map's orientation is
set by wherever the robot happened to be pointing at startup, so an
axis-aligned bounding box reports the diagonal extent of a rotated rectangle
instead of its sides — a 6.0 × 4.5 m room mapped 15° askew measures
6.9 × 5.9 m that way, and both numbers are wrong.

---

## 7. If you want to defend this in a viva

The strongest version of your project's argument is not "we used GPS and
Bluetooth". It is:

> We set out to localise with GPS and Bluetooth. We measured what each could
> actually deliver at room scale, found that GPS position error (≈ 4 m) is
> comparable to the room's own dimensions, and demonstrated quantitatively
> that naively fusing it degraded our area estimate from 27 m² to between 26
> and 54 m². We therefore restricted GPS to georeferencing and outdoor
> operation, kept Bluetooth as the telemetry link, and built the metric map
> from wheel odometry, IMU heading and range sensing — achieving 0.9 % area
> accuracy.

That is a stronger result than the original plan, and every claim in it is
reproducible from the test suite.
