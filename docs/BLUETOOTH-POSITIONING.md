# Bluetooth RSSI trilateration — implemented and measured

The project was asked to add Bluetooth positioning from signal strength.
Rather than assume it works or decline to build it, it is implemented properly
and then **measured against ground truth**. This document is the result.

---

## How it works

A BLE beacon's signal weakens predictably with distance, so a measured RSSI
can be inverted into a range:

```
RSSI = TxPower − 10 · n · log₁₀(d / d₀)
  ⟹   d = d₀ · 10^((TxPower − RSSI) / (10 · n))
```

Three beacons give three ranges, and three ranges intersect at a point. Four
over-determines it, which is solved by damped least squares
(`shared/robotmap_common/rssi.py`).

---

## The measurement

Robot driving one full circuit of a 6.0 × 4.5 m room, four beacons in the
corners, position compared against ground truth every step:

| Shadowing | RSSI error | Odometry error | RSSI ÷ odometry | Fix rate |
|---|---|---|---|---|
| 2 dB (quiet corridor) | 1.28 m | 0.07 m | 18.7× | 100 % |
| 4 dB | 1.93 m | 0.07 m | 28.2× | 100 % |
| **6 dB (typical room)** | **2.71 m** | **0.07 m** | **39.5×** | 100 % |
| 8 dB | 3.67 m | 0.07 m | 53.5× | 100 % |
| 10 dB (cluttered office) | 4.79 m | 0.07 m | 69.8× | 99 % |

**In a typical room, RSSI is about 40× less accurate than wheel odometry**, and
its error is 60 % of the room's shorter dimension.

### Why it is that bad

The inversion is exponential, so RSSI error becomes distance error very fast.
At a 2.5 path-loss exponent:

| RSSI error | Distance error at 5 m |
|---|---|
| 1 dB | ~10 % (0.5 m) |
| 3 dB | ~32 % (1.6 m) |
| 6 dB | ~74 % (3.7 m) |

Indoor shadowing is routinely 4–8 dB one sigma. **Metre-level error is the
expected outcome of a correct implementation, not a bug to be tuned away.**

---

## Why it still earns its place

The comparison above is not the whole story. Odometry and RSSI fail in
completely different ways, and that difference is the reason to keep both:

| | Over one circuit | Over an hour | Robot picked up and moved |
|---|---|---|---|
| **Odometry** | 0.07 m | drifts without bound | never notices |
| **RSSI** | 2.71 m | **still 2.71 m** | recovers immediately |

Measured across the circuit, RSSI error was **2.66 m in the first third and
2.70 m in the last** — flat. Odometry went from 0.04 m to 0.11 m and would keep
climbing.

So RSSI is the wrong tool for drawing a room and the right tool for three
things odometry cannot do at all:

- **Room-level presence** — which room the robot is in, reliably
- **Bounding drift** on long runs
- **Kidnap recovery** — a robot lifted and put down elsewhere

## What this means architecturally

RSSI is gated exactly as GPS is: it may inform *where the robot is*, but it
may not *redraw the map*. A position uncertain by 2.7 m would put walls 2.7 m
from where they belong in a room only 4.5 m across.

This is the same argument already established for GPS
([LOCALIZATION.md](LOCALIZATION.md)), at a smaller scale — 2.71 m instead of
4 m, but against the same 4.5 m room.

---

## Installing beacons: the one thing that matters

Beacon **placement** matters as much as signal quality, and it is the only
part an installer controls.

**Put them in the corners.** Corners maximise the spread of bearings from
anywhere inside, which keeps the geometry well conditioned.

**Never put them all along one wall.** Four beacons in a straight line make
trilateration *mathematically impossible*, not merely worse: any point and its
mirror image across that line fit the ranges equally well. The solver refuses
rather than guessing — `test_beacons_on_one_wall_cannot_fix_a_position_at_all`
pins that it fails loudly instead of silently halving the accuracy.

**Use four, not three.** A person standing in front of a beacon is routine;
the fourth means the fix survives it.

**Calibrate TxPower per beacon.** `TxPower` is the RSSI at exactly one metre,
and it varies by several dB between supposedly identical beacons. That error is
*systematic*, so averaging more samples does not remove it — only measuring
each beacon does. Averaging does help with shadowing, which is zero-mean.

---

## If you need better than metres indoors

| Technology | Indoor accuracy | Cost | Notes |
|---|---|---|---|
| BLE RSSI (this) | 1–5 m | ~RM 25/beacon | what is built here |
| BLE AoA (Bluetooth 5.1) | 0.1–0.5 m | ~RM 400+/anchor | needs antenna-array anchors |
| UWB (DW1000) | 0.1–0.3 m | ~RM 400 for four | the honest answer to indoor radio positioning |
| Wheel odometry + IMU | 0.07 m over a circuit | already fitted | drifts on long runs |

For this project's job — measuring one room — odometry already wins by 40×.
UWB would be the upgrade if radio positioning genuinely had to carry the map.

---

## Where the code is

| | |
|---|---|
| `shared/robotmap_common/rssi.py` | path loss, trilateration, uncertainty |
| `simulator/virtual_robot.py` | beacon simulation with shadowing, TxPower spread, body blocking |
| `tests/test_rssi.py` | 32 tests — maths, geometry, divergence |
| `tests/test_rssi_accuracy.py` | 14 tests — the measurements in this document |
| `omniverse/kit_room_3d.py` | the 3D room. The RSSI marker was removed with the fusion |

### One bug worth recording

The first implementation used plain Gauss-Newton. When noisy ranges are
mutually inconsistent the circles have no common intersection, the Jacobian
goes near-singular, and the step is unbounded — it reported positions **tens of
kilometres away**, at a mean error of 34 km, while still marking them
converged.

A wildly wrong fix that presents as valid is worse than no fix. Now
Levenberg-Marquardt damped, step-limited, and bounded to the beacon field.
`test_the_solver_never_runs_away` pins it.
