# RoomScan — a robot that measures rooms

**"How many square metres is this room?"** — answered by driving a robot around
it instead of crawling about with a tape measure.

Drive one lap. Get a floor plan, a measured floor area, and a spreadsheet to
quote from. Saved, graded for reliability, and exportable.

```
Room 1     26.77 m²    5.95 × 4.50 m    GOOD    99% observed
```

Built for the people whose money depends on that number — flooring installers
quoting per m², cleaning contractors pricing per visit, facilities managers
reporting space. See **[docs/PRODUCT.md](docs/PRODUCT.md)** for who it is for,
what the MVP covers, and what it honestly does not do yet.

Underneath: a **three-wheel holonomic (kiwi drive)** robot — three omni wheels
at 120°, driven by daisy-chained Feetech STS3215 bus servos over USB, with no
microcontroller.

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

## The whole twin, one command

```
START-TWIN.bat
```

Brings up the stack, launches Omniverse with the scene, and opens both browser
windows. `STOP-TWIN.bat` closes everything again, Omniverse included — it holds
the GPU at full tilt while a scene is open. Or `docker compose up --build -d`
for the containers alone.

### Three windows, three jobs

| | |
|---|---|
| **Omniverse** | **the physical world.** Not a picture of it — drag a table across the viewport and the robot bumps into it where you put it |
| <http://localhost:8080> | **the robot's own 2D map**, built from what it has driven over and run into |
| <http://localhost:3001> | **the data** — Grafana, admin / admin |

Each does the thing it is best at. An RTX renderer beats a browser canvas at
showing a room; a flat plan beats an extruded outline seen in perspective at
showing a floor plan; a time-series database beats both at showing how the
estimate converged. There used to be a `/twin` page drawing 3D in the browser
beside the 2D map, and it lost on both counts.

The join between them is the point. Rearranging the 3D scene rearranges the
room the robot drives in, so a contact appears on the 2D map where you moved
the furniture to, and the contact count moves in Grafana.

| also | |
|---|---|
| <http://localhost:8080/scans> | saved scans — the product loop |
| <http://localhost:8080/api/hardware> | what sensors are fitted, and what that lets the robot do |
| <http://localhost:8086> | InfluxDB (admin / roommapper123) |

A scan **saves itself** the moment the robot completes a lap, so finishing a
room and then losing it because nobody pressed a button cannot happen. Scans
live in `./scans/` on the host — a `docker compose up --build` does not touch
them.

> Grafana is on **3001**, not 3000 — the SmartClean Twin project usually holds
> 3000. Change it with `GRAFANA_PORT` in `.env`.
>
> Copy `.env.example` to `.env` and change the credentials before running this
> anywhere but your own machine.

Comes up with the simulator running, so there is data flowing and nothing to
configure. Grafana's datasource and dashboard are provisioned — no clicking.

Verified end to end in Docker: **26.77 m² measured against 27.00 m² truth,
IoU 0.993, grade EXCELLENT.**

### Or without Docker

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

### Bluetooth trilateration — built, and measured

RSSI positioning from four corner beacons is fully implemented. Measured over
a circuit of a 6.0 × 4.5 m room:

| | Error | vs odometry |
|---|---|---|
| Quiet corridor (2 dB) | 1.28 m | 18.7× worse |
| **Typical room (6 dB)** | **2.71 m** | **39.5× worse** |
| Cluttered office (10 dB) | 4.79 m | 69.8× worse |

So it is gated out of the map for the same reason as GPS — 2.71 m of error in
a 4.5 m room puts walls nowhere near where they belong.

**But it earns its place**, because it fails differently: RSSI error was 2.66 m
in the first third of the circuit and 2.70 m in the last — **flat**. Odometry
drifts without bound and cannot detect a robot being picked up and moved. RSSI
recovers instantly.

Details, including beacon-placement rules: **[docs/BLUETOOTH-POSITIONING.md](docs/BLUETOOTH-POSITIONING.md)**

---

## Moving the robot

Two paths, in the order Shebaro recommended — Omniverse first, real robot
second. Both run the **same** kinematics module, so nothing verified in
simulation has to be reimplemented for hardware.

### Omniverse

| You have | Run | Physics? |
|---|---|---|
| **Any Kit app** | paste [`omniverse/kit_room_3d.py`](omniverse/kit_room_3d.py) — **the room, and the robot driving in it** | no |
| Omniverse Kit / Code / USD Composer | [`omniverse/kit_holonomic.py`](omniverse/kit_holonomic.py) — movement only | no |
| Isaac Sim | `isaac-sim/python.bat omniverse/run_isaac.py --mode teleop` | yes |

`kit_room_3d.py` builds the digital twin scene: four solid walls, a shut door and
window, table, chairs, sofa, cabinet, and the four BLE beacons in the corners.

It draws **two** robots, and the gap between them is the point:

| | |
|---|---|
| 🔵 solid blue | where the robot actually is |
| 🟢 green ghost | where its own dead reckoning thinks it is |

That gap **is** the drift, and watching it open up over a run is the most
useful thing the scene shows: it is why the map comes back larger than the
room, and why a long contact-only scan grades poorly. The 2D map prints the
same figure in metres, so the two views explain each other rather than merely
disagreeing.

A third marker used to show where Bluetooth thought the robot was. It went with
the fusion that fed it — BLE measured *worse* than dead reckoning, 0.51 m of
error becoming 1.42 m, so nothing uses it and drawing it implied otherwise.

The scene also used to build the room the robot drew, on a pad beside the real
one. That is gone too: a flat floor plan reads better in a browser than an
extruded outline seen in perspective, and the 2D map now shows it beside the
live grid. Omniverse draws the physical room; the browser draws the
measurement; each does what it is best at.

Needs the mapper running, so there is a robot to follow:

```bash
python services/mapper/main.py --source sim --room furnished
```

Without it the left room still builds and runs — no mapper is not an error.

Kit's Python is not this project's, so the whole scene uses **only the standard
library** (`urllib`, `json`). Nothing to pip-install into Omniverse, which is
where a demo usually stops being reproducible. The geometry is covered by 23
tests in `tests/test_kit_measured_room.py`, which load the shipped file against
stub USD modules — Kit itself cannot run in CI.

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

## Two passes, because one is not enough

A scan runs in two phases, and they measure different things.

**1. The perimeter lap.** Wall-following, hugging the boundary all the way
round. Close range and good incidence angles are what an outline needs, and
this is what produces the room area. It learns nothing whatsoever about the
middle of the room.

**2. The row-by-row sweep.** A boustrophedon pass over the interior — along a
row, sidestep across, back along the next. Slower, and the only way to find a
table standing in open floor.

The outline is measured and saved after phase 1, so a sweep that is
interrupted still leaves a usable measurement behind. Phase 2 then adds what
it found to that same scan rather than creating a second one.

Three details in the sweep are not obvious and are each there for a measured
reason:

- **It about-faces between rows rather than reversing.** The base is holonomic
  and *could* drive the next row backwards, but every range sensor faces
  forward and nothing watches the rear. Holonomy still earns its keep in the
  sidestep: the robot translates to the next row without rotating, so the row
  spacing is exactly what was commanded. A differential base has to arc
  across, and the arc is what makes its rows drift out of parallel.
- **The turn is closed on the measured heading, not run for a fixed time.** An
  open-loop about-face falls short by one control step every row; six rows
  later the sweep is visibly fanning out instead of covering the floor.
- **The sensors, not the map, decide when the sweep is done.** The room bounds
  handed over by phase 1 are expressed in the pose estimate's own frame, so
  they drift with it over a long sweep. A wall actually alongside the robot
  does not drift.

`/api/coverage` reports the sweep live: rows done, collisions, and every
obstacle the robot has remembered.

### Two floor areas, not one

| | |
|---|---|
| **Total floor** | includes the space under the table — it still has to be floored |
| **Blocked** | what furniture stands on, hatched in red on both maps |
| **Usable floor** | what a cleaning robot could actually reach |

A wall and a table leg look identical to a range sensor; both are just
"occupied". What separates them is where they sit — a wall encloses the room,
furniture is surrounded by the room's own floor.

One subtlety worth stating, because it caused a wrong answer the first time: a
range sensor only ever sees an object's *outer faces*, never its middle, so a
1 m table appears as a hollow 5 cm outline and measures 0.19 m² instead of
about 1 m². Blocked floor is therefore the room's filled footprint minus its
observed free floor, not the occupied cells.

## The product loop

**Scan → judge → save → export.** A demo stops after the first step.

Every scan is graded before it is saved, because a measuring tool that is
sometimes wrong and never says so is worse than no tool:

| Grade | Meaning |
|---|---|
| **GOOD** | boundary closed, ≥85 % of floor observed, pose confident |
| **ACCEPTABLE** | closed, ≥60 % observed — usable with a wider margin |
| **POOR** | closed but thin evidence; re-scan |
| **UNUSABLE** | boundary never closed — the area is a **lower bound**, not a measurement |

Two places that grade does real work:

- The **CSV total sums only usable scans.** Adding up figures already flagged
  unreliable would give a confident-looking total made of bad parts.
- The **exported floor plan carries the warning on the drawing.** A printed
  plan outlives the app; without it a bad scan becomes a piece of paper that
  looks authoritative.

Exports: dimensioned **SVG** floor plan, **JSON** for other software, and
**CSV** of every room — which is where a quote actually gets written.

```bash
curl http://localhost:8080/api/scans.csv
```

## The digital twin view

`/twin` puts both halves side by side, which is what a digital twin actually
is — the thing, and the model of the thing:

| Left: **PHYSICAL** | Right: **DIGITAL** |
|---|---|
| the room that exists, in 3D | what the robot has worked out |
| walls, doorway, furniture, BLE beacons | occupancy grid it built from scratch |
| drag to orbit, scroll to zoom | its path, its room outline, its uncertainty |

The 3D runs **in the browser with no library and no CDN** — a small
painter's-algorithm renderer, because a lab machine with no internet still has
to be able to show it, and pulling in three.js for twenty boxes is more risk
than it saves.

Three robots are drawn in the 3D pane, and the gap between them is the point:
blue is where the robot *is*, green is where **odometry** thinks it is (0.07 m
error), orange is where **Bluetooth** thinks it is (2.71 m error).

The footer compares measured area against true area live. In simulator mode
the left pane is genuine ground truth so the error figure is exact; against
real hardware it says so, because the true room is then an assumption rather
than a measurement.

## The two screens

`/api/compare` scores one against the other:

| Screen 1 | Screen 2 |
|---|---|
| The room that actually exists | The room the robot drew from its own sensors |

Both are rendered at the **same scale**, so the shapes are comparable by eye,
and the robot's outline is overlaid faintly on screen 1 so any discrepancy
shows up in place rather than only as a number.

It is scored, not just shown:

| Metric | What it catches that the others miss |
|---|---|
| Area error | the headline — and the easiest to satisfy with a wrong map |
| Dimension error | a room the right area but the wrong proportions |
| **IoU** | size, shape *and* position together — the honest single number |
| Centroid offset | separates "wrong shape" from "right shape, wrong place" |

Why IoU rather than grading on area: a room measured 9 × 3 m has *exactly* the
correct 27 m² area while being entirely the wrong shape. Area error calls that
perfect; IoU calls it POOR. `test_area_can_be_right_while_the_grade_is_not`
pins it.

## Architecture

```
        ┌──────────────────────────────┐
        │  Robot (ESP32)               │
        │  encoders · IMU · sonar · GPS│
        └───────┬──────────────────────┘
                │ MQTT
                ▼
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
│   ├── kit_room_3d.py          Real room + the robot's own map, side by side
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
│   ├── pilot/                  Autonomous scan on real hardware, with limits
│   ├── localization/fusion.py  Odometry + IMU + gated GPS → pose
│   ├── mapping/                Log-odds grid → room polygon + area
│   └── mapper/                 Pipeline + live web UI
├── autonomy/                   Where the robot decides to drive
│   ├── bump_explorer.py        Contact-only: what THIS robot can actually run
│   ├── explorer.py             Wall-following: needs range sensors
│   └── coverage.py             Row-by-row sweep: needs range sensors
├── simulator/                  Virtual robot and world, for hardware-free dev
├── tests/                      1076 tests
└── docs/
    ├── OMNIVERSE.md            Movement: Omniverse then real robot
    ├── HARDWARE.md             BOM, wiring, calibration
    └── LOCALIZATION.md         Why GPS is gated, with measurements
```

---

## Running with real hardware

**There is no microcontroller.** Three Feetech STS3215 12 V bus servos are
daisy-chained to a servo bus board; the board takes 12 V from an adapter and
speaks to the PC over USB-C. Everything else — odometry, mapping, autonomy —
runs on the PC. See [docs/HARDWARE.md](docs/HARDWARE.md).

**There are no range sensors either**, which decides which autonomy can run:

| Strategy | Needs | On this robot |
|---|---|---|
| `bump_explorer.py` | contact only | ✅ runs today |
| `explorer.py` (wall-following) | side + front ranges | ❌ cannot run |
| `coverage.py` (row-by-row sweep) | forward + side ranges | ❌ cannot run |

The range-sensor path is kept deliberately, not by neglect: it is the
measured upgrade case. Same room, same code, the only difference being one
sensor — **1.4 % error with ranges, ~10 % without**. That number is what a
decision to fit a sensor should rest on.

**1. Find the bus and check the wiring.**

```bash
python services/servo-bus/scan.py --list
```

```bash
python services/servo-bus/calibrate.py --port COM5 --check-order
```

Run `--check-order` before anything drives itself. The servo IDs must be in the
same order as the wheel angles; get it wrong and the robot drives off at the
wrong angle while looking perfectly healthy.

**2. Start a broker.**

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto
```

**3. Start the mapper.**

```bash
python services/mapper/main.py --source mqtt
```

**4. Scan the room.** Dry-run it first — this prints the commands it would send
and touches no hardware:

```bash
python services/pilot/main.py --dry-run
```

```bash
python services/pilot/main.py --servo-port COM5
```

The pilot runs the same `WallFollower` and `CoveragePlanner` that CI tests
against the virtual robot. That is the point of keeping them in `autonomy/`
with no simulator types in their signatures: a passing test says something
about the hardware.

**Bluetooth instead of USB?** Not available. That path needed something on the
robot generating `SensorPacket` JSON to send over a serial link, and with no
microcontroller nothing does — the packets are assembled on the PC, which is
already the far end of the link. `services/bt-bridge` was deleted rather than
left in the README as a workflow that cannot work; `git log` has it if a
tetherless build ever revives the idea.

Bluetooth is still used, for the BLE *beacons* that give a bounded position
fix — a different thing entirely. See
[docs/BLUETOOTH-POSITIONING.md](docs/BLUETOOTH-POSITIONING.md).

### Before you let it drive itself

An autonomous robot with 12 V servos fails physically, so the safety behaviour
is specified rather than assumed, and tested in `tests/test_pilot.py`:

- **The wheels stop when the pilot does** — every exit path, exception and
  Ctrl-C included.
- **Stale telemetry stops the robot.** A frozen world model while the robot
  keeps moving is exactly when it hits something.
- **The driver's watchdog is the backstop.** It halts the wheels if no command
  arrives, so even `kill -9` stops the robot within `watchdog_s`.
- **Touching something draws it.** A bumper contact is written straight into
  the occupancy grid as blocked floor, so it shows up as a red patch on the 2D
  map, in the browser 3D twin, and in Omniverse — then the scan carries on. An
  obstacle is information, not a failure.

  This matters because the range sensors miss a lot of real furniture: a chair
  leg narrower than the ultrasonic beam, a sofa that absorbs the pulse,
  anything angled enough to reflect the echo away, and everything below the
  sensor's mounting height. The bumper is what catches those.

  Three things had to be true for it to work, and none were obvious:

  * **A contact is assigned, not accumulated.** A cell in a well-observed room
    sits at the log-odds clamp, and adding one contact's worth of evidence to
    −6.0 leaves it at −3.06 — still firmly "free", so the object the robot is
    touching never appears. It only ever worked for objects the sonar had never
    seen. Direct evidence supersedes inference rather than averaging with it.
  * **The free circle under the robot must not scrub it out.** That circle is
    slightly wider than the chassis and the bumper sits just outside it, so a
    robot stopped against an obstacle erased the contact within about half a
    second at 10 Hz.
  * **A touched object counts without being circled.** Obstacles are otherwise
    found by hole-filling, which needs free space observed all the way around
    them; a single touch-and-retreat never encloses anything. Measured on a
    slim pillar the sonar kept missing: 2 contacts recorded, 0.00 m² reported.
    Contacts are now tested against the room outline instead — inside it is
    furniture, on it is wall.
- **Scan again** (button on the map and twin pages, or `POST /api/rescan`)
  wipes the map and drives the whole scan from the start, repeating the run it
  was configured with rather than a default one — two scans of "the same room"
  are only comparable if both were driven the same way. A finished scan stays
  in the library as its own entry; an unfinished one is discarded.

  The previous simulator is stopped **and joined** before the map is cleared.
  Two of them interleaving packets into one pipeline would draw a room from two
  robots standing in different places, and that looks like a mapping bug rather
  than like two scans fighting. Concurrent presses get `409 busy` instead of a
  second robot.
- **A bump backs the robot off; repeated bumps halt the scan.** Halting on the
  first contact was the original behaviour and it made the robot useless — run
  against a furnished room it clipped a cabinet 43 s in and gave up having
  measured nothing. Furniture against a wall is the normal case. Six contacts
  in one lap is not, and means a robot wedged somewhere shoving.
- **It sidesteps out of a squeeze.** Wall-following regulates one wall and
  watches ahead; nothing watches the other flank, so the robot drove into the
  0.40 m gap between a cabinet and the wall reporting 4 m of clear space ahead
  the whole way in. Because the base is holonomic it translates out sideways
  *while holding its heading*, so the wall-following loop is never disturbed —
  a differential robot would have to turn away and re-acquire the wall.
- **Speed is limited in the pilot *and* in the driver**, independently, because
  the one that matters is whichever is lower.

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

## Adding a sensor

What the robot has is described in one place —
[`shared/robotmap_common/hardware.py`](shared/robotmap_common/hardware.py) —
and the mapping strategy is *derived* from it rather than configured:

```
no RANGE fitted   ->  CONTACT_ONLY     find the room by driving into it
RANGE fitted      ->  WALL_FOLLOWING   hold a measured distance to a wall
```

Set `fitted=True` on a device and the stack changes behaviour without another
line being edited. Fitting a lidar switches the robot from 435 m of bouncing at
7.9 % error to 23 m of wall-following at 1.4 %.

| | provides | fitted | why it matters |
|---|---|---|---|
| Feetech STS3215 bus | odometry, **contact** | yes | contact without a bumper: a wheel at 3 % of commanded speed and 90 % load has hit something |
| GNSS | position | yes | anchors the map outdoors; useless indoors, which is the case that matters |
| BLE beacons | position | **no** | wired up and switched off — it made the pose *worse*, 0.51 m → 1.42 m |
| 2D lidar | range | no | the single biggest upgrade available |
| Ultrasonic ring | range | no | the cheap route to range; ±15° cone is blind to a low bin |
| mmWave radar | range | no | good obstacle backstop, poor sole mapper |
| IMU | heading | no | helps most during turns, where wheel odometry is weakest |
| Arduino bridge | range, contact | no | not a sensor — a way to attach ones needing hard real-time pins |
| FPGA encoder front end | odometry | no | only if the bus turns out to drop counts; measure that first |

Every `accuracy` field is measured on this project, not quoted from a
datasheet. `--hardware actual` runs the robot that exists, and stops the
simulator handing it range readings it will not have.

<http://localhost:8080/api/hardware> reports the lot, so "why is it bouncing
off the walls instead of following them?" has an answer the dashboard can show.

---

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests/ -q
```

601 tests, about 16 seconds, no hardware, broker or Isaac Sim needed. CI runs
them on Python 3.11 and 3.12, plus a Docker build that boots the image and
checks it actually maps a room.

| File | Covers |
|---|---|
| `test_holonomic.py` | Kiwi-drive kinematics: IK/FK round trip, arc integration, strafing |
| `test_comparison.py` | Twin fidelity: IoU, alignment, and the metrics' blind spots |
| `test_twin_control.py` | Twin fan-out, slip detection, divergence bookkeeping |
| `test_nmea.py` | GPS sentence parsing, checksums, the indoor-fix case |
| `test_robot_agent.py` | PC-side packet assembly, stale-fix rejection |
| `test_drive_controller.py` | Isaac controller against a fake articulation |
| `test_source_hygiene.py` | Every file compiles, LF endings, no mojibake or BOM |
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
| `GET /api/room` | Room polygon, area, dimensions, closure, obstacles, blocked area |
| `GET /api/coverage` | Sweep progress, collisions, remembered obstacles |
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

- **A heavily furnished room is under-measured, badly.** This is the biggest
  one. Measured on a 27 m² room with a table, four chairs, a sofa, a cabinet
  and a bin:

  | strategy | result |
  |---|---|
  | wall-following (needs a lidar) | **12.89 m²** of 27.0 |
  | contact-only (the actual hardware) | **0.00 m²** — extraction fails outright |

  The wall-follower loses the wall when furniture blocks it and follows the
  furniture instead; the contact explorer covers the floor but leaves free
  space too fragmented for the extractor to close a boundary. An empty room
  measures to 1.4 %, so the demo is sound — but "map any room" is not true yet.
  Requiring the lap to have wound a full turn before closing
  (`min_loop_winding_deg`) fixed a 3.18 m hook that was closing the boundary at
  6.9 % coverage, and improved this case from 10.19 m², but did not solve it.

  The scan grade **does** catch it, so the number is never presented as
  trustworthy. A perimeter lap goes round the room, so the distance driven is
  compared with the perimeter of the outline it produced:

  | | driven | reported perimeter | ratio | grade |
  |---|---|---|---|---|
  | empty room | 18.7 m | 20.9 m | **0.90** | GOOD |
  | furnished | 10.1 m | 21.9 m | **0.46** | POOR |

  Both verified on the live stack. Contact-only mapping has no perimeter phase,
  so it is not judged on a lap it never drove.

  Contact-only on a furnished room used to report **0.00 m²** — the flood fill
  seeded on a stranded single free cell. It now returns 13.57 m² with
  `closed=False`. Still short of the truth, and honestly so.

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
- **The sweep costs accuracy to buy coverage.** The interior pass adds roughly
  50 m of driving to a 6 × 4.5 m room, and dead reckoning drifts over all of
  it. That is why the outline is kept from the perimeter lap when the sweep
  grades worse — each pass is used for what it actually measures well, and the
  grade reported is the one that outline earned.
- **Furniture pushed against a wall is counted as wall.** The wall/furniture
  distinction is "enclosed by the room's own floor", so a cabinet flush to a
  wall is not enclosed and does not appear as blocked area. It is a real gap,
  not a bug in the implementation: separating the two needs a height sensor,
  which this robot does not have.
- **Obstacle memory drifts with the pose.** Obstacles the sweep remembers are
  stored in the pose estimate's frame, so on a long sweep the same table can
  be recorded more than once. The map's obstacles come from the occupancy grid
  and do not have this problem; `/api/coverage` is the robot's own account and
  does.
