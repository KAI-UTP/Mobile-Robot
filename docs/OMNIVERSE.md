# Moving the robot: Omniverse first, then real

Following Shebaro's advice — *"first deal with knowing how to move on
Omniverse, then deal with irl movement"* — this document covers both, in that
order.

---

## The platform, restated

The robot is a **three-wheel holonomic base**, also called a kiwi drive: three
omni wheels at 120°, all driven. This is *not* a differential drive, and the
difference is not cosmetic.

| | Differential (2 wheels + caster) | **Holonomic (3 omni wheels)** |
|---|---|---|
| Degrees of freedom | 2 (forward, turn) | **3 (x, y, turn)** |
| Move sideways | must turn first | **directly** |
| Move and turn at once | limited | **fully independent** |
| Odometry quality | good | **worse — rollers slip by design** |
| Control input | left/right wheel speed | body twist (vx, vy, ω) |

The maths lives in [`shared/robotmap_common/holonomic.py`](../shared/robotmap_common/holonomic.py)
and is covered by 39 unit tests. **The same module drives Isaac Sim and the
real servo bus**, so anything verified in simulation is verified in the code
that will run on hardware.

### The one equation

Wheel *i* mounted at angle `a` from the robot's forward axis, at radius `R`:

```
v_i = -vx·sin(a) + vy·cos(a) + ω·R
```

Everything else — inverse kinematics, odometry, the Isaac controller, the
servo driver — is built on this line. It comes from projecting the wheel's
contact-point velocity onto its rolling direction; the rollers absorb the
perpendicular component, which is both what makes the robot holonomic and what
makes its odometry worse.

---

## Part 1 — Movement in Omniverse

You weren't certain which Omniverse you're running, so **both are provided**.
The kinematics are identical; only the layer that applies motion differs.

### Which one do I have?

| If you can... | You have | Use |
|---|---|---|
| Open *Window > Script Editor* in Omniverse Code/Kit/USD Composer | **Kit** | `omniverse/kit_holonomic.py` |
| Launch "Isaac Sim" from the Omniverse Launcher, or `pip install isaacsim` | **Isaac Sim** | `omniverse/run_isaac.py` |

Your DT project (`smartclean-twin/omniverse/`) uses the Kit Script Editor
approach, so if you're continuing on that machine, start there.

### Option A — Omniverse Kit (no physics)

Open *Window > Script Editor*, paste
[`omniverse/kit_holonomic.py`](../omniverse/kit_holonomic.py), press
**Ctrl+Enter**.

#### Launching Kit from the command line

If you have the Kit SDK rather than a Launcher app — packman leaves a kernel at
`C:\packman-repo\chk\kit-kernel\<version>\` — you can start it with a scene
already running:

```bash
kit.exe apps/omni.app.full.kit --exec C:\kitscene\kit_room_3d.py
```

> **`--exec` cannot take a path containing spaces.** Kit splits the argument at
> the first space and reports `Can't find a file to execute: D:/UTP/00` for a
> script under `D:\UTP\00 Reseach Project\`. Quoting does not help. 8.3 short
> names are usually disabled on non-system drives, so `GetFile().ShortPath`
> hands back the long path unchanged and fails the same way.
>
> Copy the script somewhere without spaces and run that. It only uses the
> standard library, so a copy behaves identically:
>
> ```bash
> cp omniverse/kit_room_3d.py /c/kitscene/kit_room_3d.py
> ```
>
> Pasting into the Script Editor avoids the problem entirely.

#### Both halves must come from the same robot

The scene reads the robot's position from `POSE_FILE_PATH` and the measured
room from `MAPPER_URL`. Nothing links them, so it is entirely possible to watch
a robot from one mapper beside a room measured by another — which happened
here: the robot followed a furnished-room mapper on port 8082 while the room
beside it was read from a rectangular-room mapper on 8080, showing 27.97 m² and
no obstacles next to a robot that had measured 23.37 m² and found one. A twin
whose halves disagree is worse than no twin.

Run one mapper and point both at it:

```bash
python services/mapper/main.py --source sim --room furnished --port 8083
MAPPER_URL=http://localhost:8083 kit.exe apps/omni.app.full.kit --exec C:\kitscene\kit_room_3d.py
```

> **Docker holds the port after the container stops.** `docker compose stop
> mapper` leaves `wslrelay` listening on 8080 and forwarding to a container
> that is gone, so a local mapper started on the same port binds without error
> and every request times out. Use `docker compose down`, or a different port.

The robot builds itself a room and runs through a movement sequence: forward,
strafe left, strafe right, rotate in place, diagonal, then translate *while*
rotating. A yellow nose marker shows which way it's facing — without it,
strafing and driving look identical and the demo loses its point.

Drive it manually from the console:

```python
drive(vx=0.2, seconds=3)        # forward
drive(vy=0.2, seconds=3)        # strafe LEFT without turning
drive(vx=0.2, omega_dps=45)     # translate and rotate together
stop_demo()
```

**What this does and doesn't tell you.** It computes where the robot should be
and puts it there. No friction, no slip, no mass. It is excellent for checking
the kinematics and the wall-following logic, and useless for asking whether the
robot will slip on carpet. That second question needs physics.

### Option B — Isaac Sim (with physics)

```bash
isaac-sim/python.bat omniverse/run_isaac.py --mode teleop
```

Three modes:

| Mode | Purpose |
|---|---|
| `teleop` | Keyboard control. **Start here.** |
| `verify` | Self-check: commands each axis, compares against what physics did |
| `mapping` | Autonomous wall-following, publishing to MQTT for the existing mapper |

Keys: `W`/`S` forward/back, **`A`/`D` strafe left/right**, `Q`/`E` rotate,
`SPACE` stop, `R` reset.

**It loads NVIDIA Kaya by default.** Kaya is Isaac Sim's built-in three-wheel
holonomic omni robot — the same configuration as yours — and crucially its omni
rollers are already modelled as individual rigid bodies. Authoring roller
physics by hand is by far the fiddliest part of simulating an omni wheel, and
Kaya has already done it. `--robot custom` builds a simple stand-in, but its
wheels are plain cylinders with no rollers, so it will *not* behave
holonomically. Use it to check the scene loads, not to study motion.

> **This code has not been executed.** Isaac Sim isn't installed on the machine
> it was written on. The movement maths beneath it has 65 unit tests, and the
> controller is tested against a fake articulation, but the Isaac glue itself is
> unverified. Expect to adjust `WHEEL_JOINT_NAMES` and the robot scale on first
> run — `--mode verify` exists to make that quick.

### Connecting Isaac Sim to the map

```bash
# terminal 1
isaac-sim/python.bat omniverse/run_isaac.py --mode mapping

# terminal 2
python services/mapper/main.py --source mqtt
```

Then open <http://localhost:8080>. The Isaac robot publishes the same
`SensorPacket` the real robot will, so the mapper, pose filter and web viewer
run unchanged.

> **One honest caveat about simulated sensors.** The range sensors in Isaac are
> raycasts: no beam cone, no specular dropout, no dropouts at all. Real
> ultrasonics have all three, and the specular problem was the single worst bug
> in this project's development — max-range returns painted "free space"
> straight through walls and turned a 27 m² room into 46 m². A map that only
> works because the simulated sensors are unrealistically good is a trap. The
> offline simulator (`--source sim`) models those defects deliberately; keep
> using it alongside Isaac.

---

## Part 2 — Real movement over the servo bus

### What Shebaro described

> *"There's no microcontroller, just a servo bus and daisy chained servo
> motors. The servo bus takes power from adapter and broadcasts the baud
> towards the type c cable to the pc so you can send inputs to wheel
> (velocity)."*

So the architecture is:

```
  PC  ──USB-C──►  servo bus board  ──daisy chain──►  wheel servo 1
                        ▲                            wheel servo 2
                   power adapter                     wheel servo 3
```

This is genuinely simpler than the ESP32 design from the previous session, and
it removes two problems outright:

- **No firmware.** The PC is the controller. Change the control loop, rerun a
  Python script — no reflashing.
- **No separate encoders.** Bus servos carry a 12-bit absolute magnetic
  encoder, 4096 counts per revolution. The differential design needed ≥360
  counts/rev and the bundled 20-slot discs were the single biggest source of
  error; that whole problem disappears here.

It also introduces one: **every command crosses a USB serial link with
PC-scheduler jitter.** A microcontroller closes its loop in a hard 1 ms; this
setup depends on Windows getting round to it. For mapping at walking pace that
is fine. It is worth stating explicitly in the report rather than leaving it
implicit.

### The servos: Feetech STS3215, 12 V — confirmed

Everything below is now specific to that part rather than a guess.

| | |
|---|---|
| Protocol | Feetech STS (`protocol="sts3215"`) |
| Default baud | 1,000,000 |
| Encoder | 12-bit absolute magnetic, **4096 counts/rev** |
| Feedback | position **and** speed, both readable |
| Max speed | ~3400 units ≈ 50 RPM no-load (12 V part) |
| Speed unit | **steps per second** |

Confirming the exact model caught three real errors in the driver, all of
which had been written to the older **SCS** convention. They are worth knowing
because each fails in a way that does not look like a protocol problem:

1. **Wheel mode is a Mode register (33), not zeroed angle limits.** SCS servos
   enter continuous rotation when both angle limits are zeroed; STS servos do
   not. An STS left in position mode acknowledges every command and then
   refuses to turn past its end stops — the wheel twitches and halts.
2. **Speed is in steps/second, not SCS speed units.** The SCS figure is about
   fifty times larger. Using it would send a speed fifty times too high, the
   servo would clamp to maximum, and every requested speed would come out as
   flat out.
3. **Torque must be explicitly enabled** (register 40). An STS powers up with
   its output free: it accepts commands, acknowledges packets, reports its
   position correctly, and does not move. That is indistinguishable from a
   wiring fault at a glance.

The `sts3215` protocol handles all three and is now the default everywhere.

> The 12 V part is the higher-speed variant; there is also a 7.4 V one. The
> protocol is identical, but the supply must match.

Because the servos report **both** position and speed, odometry comes free
from the bus — no separate encoders, and the feedback mirror mode in
`twin-control` works as designed.

### Step 1 — find the bus

Even with the model known, the port, baud and servo IDs still have to be
discovered. Rather than guess:

```bash
python services/servo-bus/scan.py --list
```

```bash
python services/servo-bus/scan.py --port COM5
```

This sweeps every protocol × baud × servo ID and reports what answers. It only
sends ping packets — no wheel can move — so it's safe with the robot on the
bench. It prints a ready-made `BusConfig` when it finds something.

If nothing replies, in order of likelihood:

1. **The bus board isn't powered from its adapter.** USB alone often powers the
   bridge chip but not the servo bus. Shebaro specifically mentioned the
   adapter, so this matters.
2. Wrong port.
3. Daisy chain plugged into the board's output rather than its input.
4. A servo family the scanner doesn't know — ask Shebaro for the model printed
   on the servo case.

### Step 2 — work out which servo is which wheel

```bash
python services/servo-bus/calibrate.py --port COM5 --spin-each
```

Two mistakes are nearly impossible to spot once the robot is moving, because in
both cases all three wheels turn smoothly and the robot glides off confidently
in the wrong direction:

- **Wrong order.** `servo_ids` must be listed in the same order as
  `wheel_angles_deg`. Servo ID 1 is not necessarily the front wheel.
- **Wrong direction.** Two of three wheels are usually mounted mirrored, so a
  positive command drives them backwards.

The tool settles both one wheel at a time, with the robot lifted. It then runs
two cross-checks:

- **Rotation check** — spinning on the spot is the only motion where all three
  wheels should turn the same way at the same speed. A wheel opposing the
  others has its `invert` flag wrong.
- **Translation check** — driving straight forward should leave the wheel at 0°
  turning *least*, because its rollers absorb that motion. If it's spinning
  hardest, the order is wrong.

### Step 3 — drive it

```python
from driver import BusConfig, ServoBusDriver
from robotmap_common.holonomic import BodyTwist, HolonomicGeometry

config = BusConfig(port="COM5", baud=1_000_000,
                   protocol="feetech", servo_ids=(1, 2, 3))

geometry = HolonomicGeometry(
    wheel_radius_m=0.029,     # MEASURE THIS
    wheel_offset_m=0.100,     # MEASURE THIS — centre to wheel contact patch
)

with ServoBusDriver(config, geometry) as robot:
    robot.drive(BodyTwist(vy_mps=0.1))   # strafe left
```

The `with` block matters: it stops the wheels on exit, including on exception.
A watchdog also stops them if no command arrives for 0.5 s, so a crashed
control loop doesn't leave the robot driving into a wall.

### What still needs measuring

Two numbers must come off the actual robot with a ruler:

- **`wheel_radius_m`** — under load, not from the datasheet.
- **`wheel_offset_m`** — chassis centre to each wheel's *contact patch*, not to
  the motor body. An error here makes the robot rotate slightly whenever it's
  asked to translate, which looks like a control bug and isn't.

---

## Part 3 — One command, two robots

This is the shape you observed in Shebaro's setup: type an instruction, the
real robot moves, and the Omniverse robot moves with it.

```bash
# terminal: drives the robot and publishes where it actually is
python services/twin-control/main.py --port COM5
```

```
# Omniverse Script Editor: paste omniverse/kit_twin_follower.py
```

Try it with no hardware first:

```bash
python services/twin-control/main.py --dry-run
```

Then type `square` — the robot drives a 4-sided square **by strafing, never
rotating**. A differential robot would have to stop and turn at every corner.
In a dry run it closes the loop to within 5×10⁻¹⁷ m, because nothing slips.

### Two ways to mirror, and why it matters

| | Omniverse shows | Divergence |
|---|---|---|
| **Command mirror** | what you *told* it to do | always zero — by construction |
| **Feedback mirror** (default) | where it *actually is*, from the encoders | real, and measurable |

Command mirroring always looks perfect, which is exactly the problem: the
simulated robot never slips, so after a minute the two have quietly diverged
and the display is confidently wrong. Feedback mirroring can look untidy —
wheels slipping, the robot lagging a command — and that untidiness is the
honest part.

The follower draws **two robots**: a solid one where the machine actually is,
and a translucent ghost where slip-free execution would have put it. The gap
between them *is* the sim-to-real gap, visible rather than buried in a log.

`report` prints the numbers; `save` writes a CSV for plotting.

> **A subtlety worth knowing about.** An earlier version advanced the ideal
> pose on the very first step while the real pose was still establishing its
> encoder baseline, leaving the two permanently one interval apart. With zero
> slip that read as a steadily growing error caused entirely by bookkeeping —
> quietly poisoning the one number the whole exercise produces.
> `test_no_bias_accumulates_between_the_two_poses` now pins it.

### If Kit can't install paho-mqtt

Some Kit builds sandbox pip. `--publish file` writes the pose to a temp file
instead, and the follower falls back to reading it. Crude, but it needs
nothing installed and works everywhere.

## How the pieces relate

```
        robotmap_common/holonomic.py      <- 42 tests, the single source of truth
                    │
    ┌───────────┬───┴────────┬──────────────────┬──────────────────┐
    ▼           ▼            ▼                  ▼                  ▼
kit_holonomic run_isaac  servo-bus/driver  twin-control    localization/fusion
 (kinematic)  (physics)   (real robot)     (both at once)     (odometry)
    │           │            │                  │                  │
    └───────────┴────────────┴──────────────────┴──────────────────┘
                          │
                   the same equation
```

Nothing reimplements the kinematics. That's deliberate: a second
implementation is a second thing to get wrong, and the two would drift apart
exactly when you most need to trust that sim matches real.

---

## Measuring the sim-to-real gap

Your DT project's `hardware-roadmap.md` already identifies this as the
research contribution, and this structure sets it up directly:

1. Command a twist in Isaac Sim, record where the robot ends up.
2. Command the *same* twist on the real robot, record where it ends up.
3. The difference is the gap — and because both ran identical kinematics, the
   gap is attributable to physics and hardware, not to two different control
   implementations.

`HolonomicDriveController.read_state()` deliberately exposes commanded *and*
measured twist side by side for this reason.
