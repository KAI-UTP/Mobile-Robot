# Hardware — bill of materials and wiring

Prices are indicative Malaysian retail (Shopee / Cytron / element14), August 2026,
in MYR. Treat them as a budgeting guide, not a quote.

---

## 0. What the robot actually is

**Read this before section 1.** The build was settled after the rest of this
document was written, and it went a different way:

| | |
|---|---|
| **Drive** | three Feetech **STS3215 12 V** bus servos, kiwi layout at 120° |
| **Bus** | one servo bus board, servos daisy-chained off it |
| **Power** | 12 V adapter into the bus board |
| **Link** | **USB-C from the bus board to the PC** |
| **Controller** | **none — the PC is the controller** |
| **Encoders** | the servos' own 12-bit encoders, read back over the bus |

Everything downstream — odometry, occupancy mapping, room extraction, the
autonomy — runs as ordinary Python on the PC. `services/servo-bus/driver.py`
owns the serial port and speaks the STS3215 protocol; `services/robot-agent/`
assembles the sensor packets; `services/pilot/` drives the scan.

There is **no microcontroller**. An earlier design had ESP32 firmware; it was
deleted rather than kept "just in case", because three separate documents had
grown a paragraph explaining that it was not used — which is a reliable sign
that code is costing more to explain than it is worth. Section 1 below is still
the comparison that led to the PC-side design, and `git log` still has the
firmware if the tetherless route is ever revived.

Consequences worth knowing:

- **The robot is tethered by USB-C.** That is fine for measuring one room and
  is the reason a Bluetooth SPP bridge exists as an alternative link.
- **Encoder resolution stopped being the limit.** A bus servo reports absolute
  position from a 12-bit encoder — 4096 counts per revolution against the 20
  the differential build assumed. Encoder quantisation dominated the old error
  budget (0.56 m of 0.69 m); it does not any more.
- **Omni-wheel slip replaced it as the limit.** The rollers that let the robot
  strafe also slide during ordinary driving, and that sliding is invisible to
  the encoders. This is why the mapping stack leans on range sensing rather
  than dead reckoning.

---

## 1. Choosing the controller

*Historical — see section 0. This is the comparison that was made before the
servo-bus route was chosen, and it still applies to a tetherless rebuild.*

Here is the comparison that matters for *this* project.

| | **ESP32** (recommended) | Arduino Uno/Nano + HC-05 | Raspberry Pi 4/5 |
|---|---|---|---|
| Price | ~RM 25 | ~RM 20 + RM 25 | ~RM 250+ |
| WiFi | built in | no | built in |
| Bluetooth | built in (BLE + classic) | via HC-05 module | built in |
| RAM | 320 KB | 2 KB | 2–8 GB |
| Speed | 240 MHz dual core | 16 MHz | 1.5 GHz quad core |
| ADC resolution | 12-bit | 10-bit | none on board |
| Hardware interrupt pins | every GPIO | 2 only | every GPIO |
| Can run the mapper itself | no | no | yes |
| Boot time | instant | instant | ~30 s |
| Power draw | ~0.5 W | ~0.2 W | ~4 W |

**Take the ESP32.** Three reasons specific to what you are building:

1. **You asked for GPS *and* Bluetooth.** The ESP32 has Bluetooth and WiFi on
   the same chip, so it can stream over BLE to your phone and MQTT to your
   laptop simultaneously, with the GPS on a hardware UART. On an Uno you would
   be juggling one hardware UART between the GPS and the HC-05.
2. **Interrupt pins.** The Uno has exactly two. Quadrature encoders on two
   wheels want four. You can work around it with pin-change interrupts, but
   it is fiddly and it is the first thing that breaks under load.
3. **RAM.** An Uno has 2 KB. A single telemetry packet in this project's
   format is around 700 bytes of JSON. That is a third of your entire memory
   for one message.

A Raspberry Pi is a fine choice if you later want to run the mapper on the
robot itself, but for a battery-powered mapping robot it is heavy, slow to
boot, and dislikes sudden power loss.

> If you have already bought an Arduino it is not needed. The servos are
> driven from the PC over USB and the sensor packets are assembled there, so a
> microcontroller in the middle would only forward what the PC already has.

---

## 2. Bill of materials

### Essential

| # | Item | Spec | Qty | ~RM | Notes |
|---|------|------|-----|-----|-------|
| 1 | ESP32 DevKit v1 | 30-pin, CP2102 | 1 | 25 | Get the 30-pin; 38-pin has a different pinout |
| 2 | **Geared motors with quadrature encoders** | JGA25-370, 12 V, 34:1, 11 PPR | 2 | 90 | **See §3 — this is the critical part** |
| 3 | Motor driver | TB6612FNG | 1 | 15 | More efficient and cooler than an L298N |
| 4 | Caster wheel | 1", metal ball | 1 | 6 | The third wheel |
| 5 | Wheels | 65 mm rubber, to fit the motor shaft | 2 | 16 | Diameter must match `config.h` |
| 6 | IMU | MPU6050 (GY-521) | 1 | 8 | For heading; see §4 |
| 7 | Ultrasonic sensor | HC-SR04 | 3 | 18 | Front, left, right |
| 8 | Chassis | Acrylic 2WD + caster kit | 1 | 25 | Or laser-cut your own |
| 9 | Battery | 2S Li-ion 7.4 V 2200 mAh + holder | 1 | 45 | Or 6×AA for a simpler start |
| 10 | Buck converter | MP1584 or LM2596, set to 5 V | 1 | 8 | Motors and logic must not share a rail |
| 11 | Wiring, headers, standoffs | — | — | 20 | |
| | | | **Subtotal** | **≈ RM 276** | |

### For the GPS requirement

| # | Item | Spec | Qty | ~RM | Notes |
|---|------|------|-----|-----|-------|
| 12 | GNSS module | NEO-6M or NEO-M8N with antenna | 1 | 35–60 | M8N sees more constellations; worth the extra |

### Optional but strongly recommended

| # | Item | ~RM | Why |
|---|------|-----|-----|
| 13 | ToF sensor VL53L0X ×3 | 45 | Replaces HC-SR04. Narrow beam, ~3 % accuracy, no specular dropout — the single biggest map-quality upgrade available |
| 14 | RPLiDAR A1M8 | 400 | Proper 360° SLAM. Overkill for this project, transformative if the budget exists |
| 15 | Bumper microswitches ×2 | 4 | Last-resort collision detection |

**Realistic total: RM 310–370** for the essential build plus GPS.

---

## 3. The encoder decision (read this one)

This is the part people get wrong, and it is worth more than every other
choice combined.

Most hobby chassis kits ship with a **20-slot slotted disc encoder**. On a
65 mm wheel that resolves 204 mm / 20 = **10.2 mm of travel per count**. The
problem is not the distance error — it is what a single count of *difference*
between the two wheels implies about rotation:

```
heading error per count = 10.2 mm / 150 mm wheelbase = 0.068 rad ≈ 3.9°
```

Almost 4 degrees of apparent turn from one count of quantisation noise.

I measured this in the project's own simulator. Driving one lap of a
6.0 × 4.5 m room, ending position error was:

| Encoder resolution | Position error after one lap | With all other noise removed |
|---|---|---|
| 20 counts/rev | **0.69 m** | 0.56 m |
| 100 counts/rev | 0.20 m | — |
| 360 counts/rev | 0.11 m | — |
| 1000 counts/rev | 0.10 m | 0.008 m |

The middle column is the total error. The right-hand column is the same test
with wheel slip, sonar noise, gyro drift and wheel-diameter mismatch all
switched off — so **0.56 m of the 0.69 m came from encoder quantisation
alone**. No filter can recover information the sensor never captured.

This is why `tests/test_integration.py::test_encoder_resolution_dominates_drift`
exists: to stop the requirement being quietly downgraded later.

**What to buy:** a motor with a magnetic quadrature encoder *on the motor
shaft*, before the gearbox, so the gear reduction multiplies its resolution:

```
counts per wheel revolution = PPR × 4 (quadrature edges) × gear ratio
JGA25-370, 11 PPR, 34:1  →  11 × 4 × 34 = 1496 counts/rev
```

That is 0.14 mm per count — 75× better than the slotted disc, for about
RM 45 more per pair. Set `TICKS_PER_REVOLUTION` in `config.h` to match.

---

## 4. Why an IMU is not optional

Wheel odometry alone cannot tell "the robot turned" from "one wheel slipped".
The IMU resolves that, and it is what keeps the heading honest.

But it only works if you **calibrate the gyro at boot while the robot is
completely still**. Every MPU6050 reports a small non-zero rotation rate when
stationary. Integrated over a mapping run, that constant becomes real error:

| Gyro bias | Heading error after a 2-minute run |
|---|---|
| 0.15 °/s (uncalibrated) | ≈ 19° |
| 0.02 °/s (after boot calibration) | ≈ 2.4° |

Nineteen degrees of drift shears a rectangular room into a shape that no
longer measures as a rectangle. The firmware already does this calibration —
`calibrateGyro()` in `sensors.h` averages 500 stationary samples — and blinks
the LED while it runs. **Do not move the robot while it blinks.**

---

## 5. Ultrasonic vs time-of-flight

The HC-SR04 is cheap and it works, but it has one property that actively
fights room mapping: a **~30° beam cone** and **specular reflection**. A pulse
striking a wall at more than about 60° of incidence bounces away instead of
returning, and the sensor reports maximum range — indistinguishable from
"nothing there".

This is not a theoretical concern. It is the failure that produced the worst
bug in this project's development: rays reported at max range painted a 4 m
line of "free space" straight through the wall, the room's boundary sprang a
leak, and the reported area of a 27 m² room came out at 46 m². The mapper now
defends against it in two independent ways (see
`services/mapping/room_extraction.py`), but the cleanest fix is better
sensors.

**VL53L0X time-of-flight** modules cost about RM 15 each, use a ~25° cone,
are accurate to ~3 %, and do not suffer specular dropout at anything like the
same rate. If you can spend RM 45, spend it here rather than on the GPS.

---

## 6. Wiring — ESP32

### Power

```
2S Li-ion 7.4V ──┬── TB6612FNG VM (motor power)
                 │
                 └── MP1584 buck ── 5V ──┬── ESP32 VIN
                                         ├── TB6612FNG VCC (logic)
                                         ├── HC-SR04 VCC ×3
                                         └── GPS VCC
```

Two rules that will save you hours:

- **Common ground everywhere.** Battery −, buck −, ESP32 GND, driver GND,
  every sensor GND. A floating ground makes sensors read plausible nonsense.
- **Never power motors from the ESP32's 5 V pin.** The current spike on
  startup browns out the regulator and the board resets. It looks exactly
  like a firmware crash and it is not.

### Signal pins

| ESP32 pin | Connects to | Notes |
|---|---|---|
| 34 | Left encoder A | **Input only** — correct for an encoder |
| 35 | Left encoder B | **Input only** |
| 32 | Right encoder A | |
| 33 | Right encoder B | |
| 25 | TB6612 PWMA | Left motor speed |
| 26 | TB6612 AIN1 | Left direction |
| 27 | TB6612 AIN2 | Left direction |
| 14 | TB6612 PWMB | Right motor speed |
| 12 | TB6612 BIN1 | ⚠ strapping pin — see below |
| 13 | TB6612 BIN2 | |
| 5 | HC-SR04 TRIG (all three) | Shared trigger |
| 18 | HC-SR04 #1 ECHO (front) | **Needs a divider — see below** |
| 19 | HC-SR04 #2 ECHO (left) | |
| 21 | HC-SR04 #3 ECHO (right) | |
| 22 | MPU6050 SDA | I²C |
| 23 | MPU6050 SCL | I²C |
| 16 | GPS TX | ESP32 receives |
| 17 | GPS RX | ESP32 transmits |
| 4 | Bumper switch | To GND, internal pull-up |
| 36 | Battery divider | ADC1_CH0 |

### Three wiring traps

**1. The HC-SR04 ECHO pin outputs 5 V. The ESP32 is 3.3 V.**
Connecting it directly will damage the pin, sometimes immediately, sometimes
after a week. Use a divider on every ECHO line:

```
ECHO ──[1kΩ]──┬── ESP32 GPIO
              │
            [2kΩ]
              │
             GND
```

That gives 5 V × 2/3 = 3.3 V. Three sensors, six resistors, about 20 sen.

**2. GPIO 12 is a strapping pin.** If it is pulled high at boot, the ESP32
selects the wrong flash voltage and will not start. The TB6612's inputs are
high-impedance so this is usually fine, but if your board refuses to boot with
the driver connected, move BIN1 to GPIO 15 and update `config.h`.

**3. GPIO 34–39 are input only.** They have no output driver and no internal
pull-up. That is exactly what you want for encoders (fit external 10 kΩ
pull-ups if your encoder is open-drain), but you cannot use them for motors.

### Sensor placement

Mount the three ultrasonic sensors at **0°, +90°, −90°** and set
`ULTRASONIC_ANGLE_DEG` in `config.h` to match. The side-facing pair is what
makes wall-following work; a front-only robot cannot hold a standoff distance.

Mount the **GPS antenna as high as possible with a clear view of the sky**,
and away from the ESP32 and the motor driver — both are strong sources of
wideband noise right where GNSS is trying to hear a very weak signal.

---

## 7. Assembly order

Build it in stages and test each one. Wiring everything and then flashing is
how you end up with three faults at once and no way to isolate them.

1. **Chassis, motors, caster.** Confirm the robot rolls freely by hand.
2. **Power.** Buck converter set to 5.0 V *before* connecting anything to it.
3. **ESP32 alone.** Flash a blink sketch. Confirm it enumerates over USB.
4. **Encoders.** Flash the firmware, open the serial monitor, turn each wheel
   by hand. Counts must rise turning forward and fall turning backward. If one
   only rises, its B channel is not wired.
5. **Motor driver.** Confirm each wheel turns the direction you commanded.
   Swap the motor leads if not — do not fix it in software.
6. **IMU.** Confirm `[IMU] found` at boot and that the reported bias is small
   (< 1 °/s). A bias of 50 °/s means the robot moved during calibration.
7. **Ultrasonic.** Hold a book at a measured 50 cm and check the reading.
8. **GPS.** Take it outdoors. Expect **5–15 minutes for a first fix** on a
   cold NEO-6M. Indoors it will very likely never fix — that is normal and
   the whole reason this project does not rely on it.
9. **Full run.** `python services/mapper/main.py` and drive a lap.

---

## 8. Calibration

Two constants must be measured, not guessed. Both live in `config.h`.

**`WHEEL_DIAMETER_M`** — mark a wheel, drive the robot straight for what the
odometry reports as 5.00 m, and measure the actual distance with a tape.

```
corrected_diameter = current_diameter × (measured_distance / 5.00)
```

**`WHEEL_BASE_M`** — command exactly ten full on-the-spot rotations and
measure how far past (or short of) the start heading it ends up.

```
corrected_base = current_base × (3600° / actual_degrees_turned)
```

Effective wheelbase is usually a few millimetres larger than the physical
measurement, because the tyres deform and scrub during a turn.

**`LEFT_WHEEL_TRIM`** — drive straight for 3 m. If the robot veers left, its
left wheel is travelling less than the encoder claims; reduce the trim by ~1 %
and repeat. Note that on the simulator this mattered far less than encoder
resolution, so fix the encoders first.
