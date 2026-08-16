# Bluetooth and GPS on the servo-bus robot

Both were in the original brief, and both had to be rethought when the platform
turned out to have **no microcontroller**. This explains where each one now
lives, and what changed.

---

## What the platform change broke

The earlier design put an ESP32 on the robot. It read the encoders, an IMU and
a GPS module, and transmitted over WiFi or Bluetooth. With that gone:

| Job | Was | Now |
|---|---|---|
| Read wheel positions | ESP32 quadrature ISR | servo bus reports them (12-bit absolute) |
| Read GPS | ESP32 UART + NMEA parser | **USB receiver on the PC** → `services/gps-reader` |
| Read IMU | ESP32 I2C to MPU6050 | **nothing — see the gap below** |
| Build telemetry | ESP32 firmware | **`services/robot-agent`** on the PC |
| Bluetooth link | ESP32 BLE | **transport to the servo bus itself** |

The message schema did not change, so the mapper, pose filter and web viewer
work exactly as before.

---

## GPS

### Where it plugs in

Into the **PC**, not the robot. A USB GNSS receiver appears as a serial port
and streams NMEA; `services/gps-reader` parses it and hands `GpsData` to the
agent.

Find it without guessing — a receiver talks unprompted, so it identifies
itself:

```bash
python services/robot-agent/main.py --find-gps
```

Then:

```bash
python services/robot-agent/main.py --servo-port COM5 --gps-port COM7
```

### Nothing about the GPS *decision* changed

The measured conclusion from [LOCALIZATION.md](LOCALIZATION.md) still holds:
consumer GPS is ~4 m accurate, a room is 4–6 m across, and letting it steer the
map moved a 27 m² room's reported area to between 26 and 54 m². The accuracy
gate (`gps_max_accuracy_for_correction_m`, default 1.0 m) is untouched. GPS
georeferences the map and supports outdoor operation; it does not draw rooms.

### What the parser had to get right

Three things, all of which produce a *plausible but wrong* answer if missed —
the most dangerous kind of bug in this project:

- **`ddmm.mmmm` is not decimal degrees.** `4807.038` means 48° 7.038′, not
  48.07°. Reading it naively puts the robot hundreds of kilometres away.
- **The checksum matters.** A truncated sentence parses into a perfectly
  well-formed position that is simply wrong.
- **Quality lives in GGA, not RMC.** RMC is the most commonly quoted sentence
  and carries only a valid/invalid flag — no satellite count, no HDOP. A
  receiver indoors emits RMC marked *valid* while being wrong by tens of
  metres. GGA drives the quality assessment here for exactly that reason.

31 tests cover this in `tests/test_nmea.py`, including the indoor case.

### Stale fixes

A fix that stops updating must not keep being reported as current — the pose
filter would treat a minute-old position as a live measurement. Since a
receiver going quiet is precisely what happens when the robot drives indoors,
this is the normal case, not an edge case. The agent drops fixes older than
5 seconds and counts them in `gps_fixes_stale`.

---

## Bluetooth

### Its role changed completely

There is no microcontroller to run BLE, so Bluetooth is no longer a telemetry
uplink. It is now **an alternative transport to the servo bus itself**.

```
 USB:        PC ──USB-C cable──► servo bus board ──► wheels
 Bluetooth:  PC ~~~BT SPP~~~~~► servo bus board ──► wheels
```

If the bus board is reachable over Bluetooth SPP, Windows exposes it as an
ordinary COM port and **nothing in the code changes** — you just point at a
different port:

```bash
python services/robot-agent/main.py --servo-port COM7 --bluetooth
```

That works because `ServoBusDriver` only ever talks to a serial port. The
`--bluetooth` flag affects nothing but the `link` field in the telemetry, so
the map viewer can show how the robot is connected.

### Whether it will work depends on the board

Two possibilities, and Shebaro's answer will settle it:

1. **The board has a Bluetooth variant or module.** Then pair it and use the
   outgoing COM port. Done.
2. **It is USB only.** Then a Bluetooth-serial adapter (HC-05 wired to the
   board's UART) can be added — but note the bus usually runs at 1 Mbaud and
   an HC-05 tops out around 460 kbaud, so the servos may need reconfiguring to
   a slower rate first.

### `services/bt-bridge` has been deleted

That service parsed `SensorPacket` JSON arriving over a Bluetooth serial link —
it assumed something on the robot was *generating* that JSON. With no
microcontroller, nothing is: the packets are assembled on the PC, which is
already the far end of the link, so bridging them to the PC is a round trip to
nowhere.

It was removed rather than kept, because a README advertising a workflow that
cannot work costs more than the code saves. `git log` has it if a tetherless
build ever revives the idea.

---

## The tether problem, and it is worth deciding early

The robot is connected to the PC by a **USB-C cable**. Its range is therefore
the length of that cable. That has real consequences for this project:

- **Room mapping** needs the robot to drive a full lap of the walls. A 6 × 4.5 m
  room has an ~18 m perimeter. A 2 m cable does not reach.
- **Outdoor GPS operation** is impossible while tethered to a desk.

Three ways out, in increasing order of effort:

| Option | Cost | Notes |
|---|---|---|
| **Laptop carried alongside** | free | Works today. Ungainly, and the cable snags — but fine for a first demo |
| **Bluetooth link to the bus** | ~RM 25, or free if the board supports it | Cuts the cable entirely. Best value if the board can do it |
| **Mini PC on the robot** | RM 250–450 | Raspberry Pi 5 or similar rides on the robot; fully autonomous. Needs its own battery |

**Worth raising with Dr**, because it changes what a demo can show. Until it is
resolved, the honest description is that the robot maps within cable reach.

---

## The real gap: no IMU

This is the one genuine capability loss from dropping the microcontroller, and
it is worth being explicit about rather than discovering later.

Heading now comes **only** from the three wheel encoders. On a holonomic base
that is a weak source: omni wheels slip sideways by design, and sideways slip
is exactly the motion that corrupts a heading estimate. The differential build
had an MPU6050 wired to the ESP32 to fix this.

The agent reports `imu=None` honestly rather than synthesising a heading from
the wheels — labelling a wheel-derived heading as an IMU reading would tell the
pose filter it had been independently confirmed when it had not, and the filter
weights IMU headings heavily.

The pose filter already widens its uncertainty for holonomic odometry
(`holonomic_noise_multiplier`, default 2.0), so the map degrades gracefully
rather than silently.

**Fix:** a USB IMU on the PC, or a small USB-serial IMU board. A BNO055 fuses
internally and outputs a stable heading (~RM 130–180). This is the single
highest-value addition to the current hardware, ahead of GPS.
