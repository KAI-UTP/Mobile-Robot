# Room Mapper — 3-Wheel Holonomic Mobile Robot

A **three-wheel holonomic (kiwi drive)** robot that drives around a room and
measures it: floor area in m², dimensions, and a floor plan you can export.
Three omni wheels at 120°, driven by daisy-chained bus servos over USB.

> **Platform note.** An earlier iteration of this repository assumed
> differential drive (two driven wheels + caster). The actual robot is
> holonomic, which is a different kinematic problem entirely — it can strafe
> sideways without turning. The holonomic maths lives in
> [`shared/robotmap_common/holonomic.py`](shared/robotmap_common/holonomic.py);
> the differential helpers in `geometry.py` remain only for the offline
> simulator and must not be used for the real robot.
> See **[docs/OMNIVERSE.md](docs/OMNIVERSE.md)**.

Built on the architecture of [SmartClean Twin](../../UG%20Y2S3/03%20Digital%20Twin/smartclean-twin)
— same MQTT topic layout, same Pydantic-contract-first approach, same
service structure — but for real hardware rather than a simulated twin.

Universiti Teknologi PETRONAS · Research Project

---

## See it work in 30 seconds

No hardware, no broker, no Docker required:

```bash
pip install -r requirements.txt
```

```bash
python services/mapper/main.py --source sim --speed 8
```

Open <http://localhost:8080>. A virtual robot follows the walls of a
6.0 × 4.5 m room while the map builds live in the browser. It finishes with:

```
Circuit complete after 23.5 m
Room: 26.77 m2, 5.95 x 4.50 m, closed=True
```

Ground truth is 27.0 m² and 6.0 × 4.5 m — **0.9 % area error**.

Try the other rooms:

```bash
python services/mapper/main.py --source sim --room l-shaped --speed 8
```

```bash
python services/mapper/main.py --source sim --room furnished --speed 8
```

---

## About GPS — read this first

You asked for GPS and Bluetooth to locate the robot and draw the room. Here is
the honest answer, measured rather than asserted:

**GPS cannot draw a room.** Consumer GNSS is accurate to about 4 m outdoors.
A room is 4–6 m across, so the error is the size of the thing being measured.
Indoors it is worse — you usually get no fix at all, or a plausible-looking
one that is wrong by tens of metres.

I measured what happens if you use it anyway. Mapping the same 27.0 m² room
five times with different noise:

| | Area reported |
|---|---|
| GPS corrections accepted | 26 – 54 m² (varies with noise alone) |
| GPS corrections rejected | 25.8 – 27.2 m² |

So the system uses each technology for what it is actually good at:

| | Role |
|---|---|
| **Wheel encoders + IMU** | Where the robot is (drives the map) |
| **Ultrasonic / ToF** | Where the walls are |
| **Bluetooth** | Telemetry link to laptop or phone |
| **GPS** | Georeferences the finished map; primary position **outdoors** only |

The filter enforces this in code: a fix may only correct the pose if its
estimated accuracy beats `gps_max_accuracy_for_correction_m` (default 1.0 m).
Consumer GPS never clears that bar, so it anchors the map and nothing more.
RTK GNSS does clear it, and is used for corrections when present.

Full reasoning and numbers: **[docs/LOCALIZATION.md](docs/LOCALIZATION.md)**

---

## Moving the robot

Two paths, in the order Shebaro recommended — Omniverse first, real robot
second. Both run the **same** kinematics module, so nothing verified in
simulation has to be reimplemented for hardware.

### Omniverse

| You have | Run | Physics? |
|---|---|---|
| Omniverse Kit / Code / USD Composer | paste [`omniverse/kit_holonomic.py`](omniverse/kit_holonomic.py) into *Window > Script Editor* | no — kinematic, like your DT project |
| Isaac Sim | `isaac-sim/python.bat omniverse/run_isaac.py --mode teleop` | yes |

Keys in teleop: `W`/`S` forward/back, **`A`/`D` strafe left/right**, `Q`/`E`
rotate. The strafe keys are the point — a differential robot cannot do that.

> The Isaac Sim scripts have **not been executed** — Isaac Sim isn't installed
> on the machine they were written on. The movement maths has 65 unit tests and
> the controller is tested against a fake articulation, but the Isaac glue is
> unverified. `--mode verify` exists to shake it out quickly.

### One command, both robots

Type an instruction, the real robot moves, and the Omniverse robot moves with
it — the digital twin loop.

```bash
python services/twin-control/main.py --dry-run
```

Then type `square`: the robot drives a 4-sided square **by strafing, never
rotating**. Paste [`omniverse/kit_twin_follower.py`](omniverse/kit_twin_follower.py)
into the Script Editor and the twin follows along, with a translucent ghost
showing where slip-free execution would have put it. The gap between solid and
ghost *is* the sim-to-real gap.

`report` prints the numbers, `save` writes a CSV.

### Bluetooth and GPS

Both are still in, but the platform change moved them. With no microcontroller,
nothing on the robot can read a GPS or speak Bluetooth — so both went PC-side:

| | Was (ESP32) | Now |
|---|---|---|
| **GPS** | module on the robot's UART | **USB receiver on the PC** → `services/gps-reader` |
| **Bluetooth** | BLE telemetry uplink | **transport to the servo bus** — just a different COM port |
| Telemetry assembly | ESP32 firmware | **`services/robot-agent`** |

```bash
python services/robot-agent/main.py --find-gps
```

```bash
python services/robot-agent/main.py --servo-port COM5 --gps-port COM7
```

Bluetooth needs no code change — point at the paired port instead:

```bash
python services/robot-agent/main.py --servo-port COM7 --bluetooth
```

The GPS *decision* is unchanged: it still georeferences and never steers the
map. Details, plus the tether/range problem and the missing IMU, are in
**[docs/BLUETOOTH-AND-GPS.md](docs/BLUETOOTH-AND-GPS.md)**.

### Real robot (servo bus over USB-C)

No microcontroller: the bus board is powered from its adapter and appears to
the PC as a serial port. The servos are **Feetech STS3215, 12 V** — 4096-count
absolute encoders reporting both position and speed, so odometry comes free
from the bus.

Find the port, baud and servo IDs (ping packets only, nothing moves):

```bash
python services/servo-bus/scan.py --port COM5
```

It sweeps every protocol × baud × ID and prints a ready-made config. Only ping
packets are sent, so no wheel can move. Then:

```bash
python services/servo-bus/calibrate.py --port COM5 --spin-each
```

## Architecture

```
        ┌──────────────────────────────┐
        │  Robot (ESP32)               │
        │  encoders · IMU · sonar · GPS│
        └───────┬──────────────┬───────┘
                │ WiFi/MQTT    │ BLE / HC-05
                │              ▼
                │      ┌───────────────┐
                │      │  bt-bridge    │ serial → MQTT
                │      └───────┬───────┘
                ▼              ▼
        ┌──────────────────────────────┐
        │  Mosquitto  roommapper/…/raw │
        └───────────────┬──────────────┘
                        ▼
        ┌──────────────────────────────┐
        │  mapper service              │
        │   ├ pose filter (fusion)     │
        │   ├ occupancy grid           │
        │   └ room extraction          │
        └───────────────┬──────────────┘
                        ▼
              live web map · REST API
```

The mapper runs off-board deliberately: the occupancy grid for a room is
hundreds of kilobytes, which does not fit alongside WiFi buffers in the
ESP32's 320 KB, and a mapping bug is then fixed by restarting a Python
process rather than reflashing a robot that is under a desk.

---

## Repository layout

```
Mobile Robot/
├── shared/robotmap_common/     Contracts and maths — the source of truth
│   ├── holonomic.py            KIWI DRIVE kinematics — the real platform
│   ├── geometry.py             Differential drive (offline simulator only)
│   ├── models.py               Pydantic message schemas
│   ├── topics.py               MQTT topic constants
│   └── mqtt_client.py
├── omniverse/
│   ├── kit_holonomic.py        Kit Script Editor — kinematic, no physics
│   ├── kit_twin_follower.py    Omniverse robot follows the real one
│   ├── run_isaac.py            Isaac Sim — physics, teleop/verify/mapping
│   ├── drive_controller.py     Body twist → wheel joints (fully tested)
│   ├── isaac_bridge.py         Isaac → SensorPacket → MQTT
│   └── isaac_compat.py         Handles isaacsim.* vs omni.isaac.*
├── services/
│   ├── robot-agent/            Replaces the ESP32 firmware: sensors → MQTT
│   ├── gps-reader/             NMEA from a USB GNSS receiver on the PC
│   ├── twin-control/           One command → real robot + Omniverse twin
│   ├── servo-bus/              REAL ROBOT: USB serial to daisy-chained servos
│   │   ├── scan.py             Identify protocol, baud and servo IDs
│   │   ├── calibrate.py        Which servo is which wheel, and which way
│   │   ├── protocols.py        Feetech / Dynamixel / LewanSoul framing
│   │   └── driver.py           Holonomic drive + watchdog
│   ├── localization/fusion.py  Odometry + IMU + gated GPS → pose
│   ├── mapping/                Log-odds grid → room polygon + area
│   ├── mapper/                 Pipeline + live web UI
│   └── bt-bridge/              Bluetooth serial → MQTT
├── simulator/                  Virtual robot + wall-following explorer
├── firmware/esp32_robot/       Superseded — kept for the microcontroller route
├── tests/                      230 tests
└── docs/
    ├── OMNIVERSE.md            Movement: Omniverse then real robot
    ├── HARDWARE.md             BOM, wiring, calibration
    └── LOCALIZATION.md         Why GPS is gated, with measurements
```

---

## Running with real hardware

**1. Build the robot.** [docs/HARDWARE.md](docs/HARDWARE.md) has the full BOM
(≈ RM 310–370), the wiring table, and three traps that will otherwise cost you
an afternoon — including the 5 V ECHO pin that damages ESP32 GPIOs.

**2. Flash the firmware.** Edit `firmware/esp32_robot/config.h` (WiFi
credentials, MQTT host, `TICKS_PER_REVOLUTION`), then flash with the Arduino
IDE. Requires the `PubSubClient` and `ArduinoJson` libraries.

**3. Start a broker.**

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto
```

**4. Start the mapper.**

```bash
python services/mapper/main.py --source mqtt
```

**Bluetooth instead of WiFi?** Find the port, then bridge it:

```bash
python services/bt-bridge/main.py --list
```

```bash
python services/bt-bridge/main.py --port COM5
```

---

## The one hardware decision that matters

Do not use the 20-slot disc encoders bundled with hobby chassis kits. On a
65 mm wheel one count is 10.2 mm of travel, and a single count of *difference*
between the two wheels reads as **3.9° of rotation**.

Measured on the simulator, one lap of a 6 × 4.5 m room:

| Encoder resolution | Position error | With all other noise removed |
|---|---|---|
| 20 counts/rev | 0.69 m | 0.56 m |
| 360 counts/rev | 0.11 m | — |
| 1000 counts/rev | 0.10 m | 0.008 m |

**0.56 m of the 0.69 m came from encoder quantisation alone.** No filter
recovers information the sensor never captured. Use a quadrature encoder on
the motor shaft, before the gearbox — an 11 PPR encoder behind a 34:1 gearbox
gives 1496 counts per wheel revolution.

`tests/test_integration.py::test_encoder_resolution_dominates_drift` pins this
so it cannot be quietly downgraded later.

---

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests/ -q
```

305 tests, about 14 seconds, no hardware, broker or Isaac Sim needed.

| File | Covers |
|---|---|
| `test_holonomic.py` | Kiwi-drive kinematics: IK/FK round trip, arc integration, strafing |
| `test_twin_control.py` | Twin fan-out, slip detection, divergence bookkeeping |
| `test_nmea.py` | GPS sentence parsing, checksums, the indoor-fix case |
| `test_robot_agent.py` | PC-side packet assembly, stale-fix rejection |
| `test_drive_controller.py` | Isaac controller against a fake articulation |
| `test_holonomic_fusion.py` | Three-wheel odometry through the pose filter |
| `test_servo_protocols.py` | Servo wire formats: framing, checksums, byte order |
| `test_geometry.py` | Differential kinematics, geodesy, polygon maths |
| `test_localization.py` | Sensor fusion, and every GPS rejection path |
| `test_mapping.py` | Occupancy grid mechanics, room extraction accuracy |
| `test_integration.py` | Full pipeline against rooms of known size |

The integration tests assert **numbers against ground truth**, not that the
code runs. A mapping system that produces a pretty picture and the wrong area
is worse than useless for a research project.

---

## API

With the mapper running:

| Endpoint | Returns |
|---|---|
| `GET /` | Live map viewer |
| `GET /api/room` | Room polygon, area, dimensions, closure |
| `GET /api/state` | Pose, trail, grid metadata, diagnostics |
| `GET /api/grid` | Raw occupancy bytes |
| `GET /health` | Service health and filter diagnostics |
| `POST /api/reset` | Clear the map and restart |
| `WS /ws` | Live pose + grid stream |

```bash
curl http://localhost:8080/api/room
```

---

## Known limits

Stated plainly, because a research project should be honest about them:

- **No loop-closure correction.** Drift accumulated over a lap is not
  redistributed when the robot returns to its start. Below ~30 m of driving
  this does not matter; over a long multi-room run it would.
- **Single room at a time.** The flood fill deliberately stops at walls, so a
  robot that drives through a doorway maps the room it started in. Multi-room
  mapping needs a topological layer on top.
- **Static world assumed.** People walking through are eventually erased by
  the log-odds decay, but they blur the map while present.
- **Sonar hates angled walls.** Beyond ~60° of incidence the pulse reflects
  away and reads as maximum range. The mapper defends against this in two
  independent ways; ToF sensors avoid it at the source.
