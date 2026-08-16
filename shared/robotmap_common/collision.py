"""Noticing that the robot has run into something, without a bumper.

The problem
-----------
This robot has no contact switch. Its whole sensor list is a servo bus, BLE
beacons and GNSS. So a collision cannot be *read*; it has to be inferred.

The obvious idea — "the reported position stopped changing" — does not work on
its own, and it is worth being precise about why, because it is the first thing
anyone suggests:

============  ==================  ==========================================
Signal        Accuracy            Can it see a collision?
============  ==================  ==========================================
BLE RSSI      2.71 m (measured)   No. A collision stops motion by centimetres
GNSS indoors  ~25 m               No
Wheel odometry derived from       No, when the wheels are still turning: the
              the servos          pose keeps advancing through the obstacle
Servo speed   per wheel, direct   YES, within a few control cycles
Servo load    per wheel, direct   YES, and it also catches spinning wheels
============  ==================  ==========================================

So the fast, reliable evidence is the servo bus, which reports what each wheel
is *actually* doing rather than what it was told to do.

Two detectors, at two timescales
--------------------------------
**Fast — the wheels are not delivering.** The robot asks for 0.18 m/s and the
servos report a fraction of it, or report it only by drawing far more load than
free running needs. That is something solid in the way. Fires in a few tenths
of a second, which is what you need to stop before scrubbing paint off a table
leg.

**Slow — the robot has not gone anywhere.** Over a long window, compare the
distance commanded against the distance actually covered. This is the "position
stopped changing" idea, applied at a timescale where it genuinely works: BLE
cannot resolve a 5 cm stall, but it *can* tell that a robot which should have
travelled three metres is still in the same place, because three metres is
comfortably outside its 2.71 m error. It is the backstop that catches what the
fast detector misses — a wheel spinning freely on a smooth floor delivers
speed and normal load while the robot goes nowhere.

Neither is a bumper, and the difference matters when reading the map: a contact
here means "the robot could not get through", which is what a red patch on the
floor plan should mean anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionConfig:
    """Thresholds for calling something a collision."""

    # Below this fraction of the commanded wheel speed, the wheel is being
    # held. Not near zero: omni rollers creep and scrub even when the chassis
    # is firmly stopped, so a blocked wheel still reports some motion.
    speed_ratio: float = 0.35

    # Normalised servo load, 0..1, above which the wheel is straining. A wheel
    # driving the robot along a floor sits well under half.
    load_threshold: float = 0.70

    # Commands slower than this are not worth judging. A robot easing to a
    # halt legitimately reports almost no wheel speed, and calling that a
    # collision would put a red patch wherever it stopped to turn.
    min_commanded_mps: float = 0.05

    # Consecutive cycles that must agree before a collision is declared. One
    # bad read is a dropped packet on a serial bus; three in a row is contact.
    # At 10 Hz this costs 0.3 s and about 5 cm of travel.
    confirm_cycles: int = 3

    # ── Slow detector ────────────────────────────────────────────────────
    # How long to watch before comparing commanded travel with real travel.
    stall_window_s: float = 8.0
    # Fraction of the commanded distance that must actually be covered.
    min_progress_ratio: float = 0.25
    # Never judge on less than this much commanded travel, so the window's
    # verdict is always well clear of the position estimate's own error.
    min_commanded_travel_m: float = 3.0


@dataclass
class CollisionEvent:
    """A detected collision, and what gave it away."""

    x_m: float
    y_m: float
    heading_deg: float
    reason: str
    detector: str            # "wheels" or "no-progress"
    wheel_index: int | None = None

    def describe(self) -> str:
        return f"{self.reason} at ({self.x_m:.2f}, {self.y_m:.2f})"


@dataclass
class _Window:
    elapsed_s: float = 0.0
    commanded_m: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    started: bool = False


class CollisionDetector:
    """Infers contact from servo feedback and from lack of progress.

    Deliberately stateful: both detectors need history. A single control cycle
    cannot distinguish a collision from a dropped serial read.
    """

    def __init__(self, config: CollisionConfig | None = None) -> None:
        self.config = config or CollisionConfig()
        self._agreeing_cycles = 0
        self._window = _Window()
        # Set while contact persists, so one obstacle is reported once rather
        # than every cycle the robot spends leaning on it.
        self.in_contact = False
        self.events: list[CollisionEvent] = []

    # ── Fast detector ─────────────────────────────────────────────────────

    def _wheels_blocked(
        self,
        commanded_wheel_speeds,
        measured_wheel_speeds,
        wheel_loads,
    ) -> tuple[bool, str, int | None]:
        cfg = self.config

        if not commanded_wheel_speeds or measured_wheel_speeds is None:
            return False, "", None

        for index, commanded in enumerate(commanded_wheel_speeds):
            if abs(commanded) < cfg.min_commanded_mps:
                continue
            if index >= len(measured_wheel_speeds):
                continue

            measured = measured_wheel_speeds[index]
            delivering = abs(measured) / abs(commanded)

            load = None
            if wheel_loads is not None and index < len(wheel_loads):
                load = wheel_loads[index]

            if delivering < cfg.speed_ratio:
                # Straining as well as slow is unambiguous. Slow on its own is
                # still reported, because a servo that has given up entirely
                # may draw little: the wheel is stopped either way.
                if load is not None and load >= cfg.load_threshold:
                    return (
                        True,
                        f"wheel {index} stalled — {delivering:.0%} of commanded "
                        f"speed at {load:.0%} load",
                        index,
                    )
                return (
                    True,
                    f"wheel {index} delivering {delivering:.0%} of commanded speed",
                    index,
                )

        return False, "", None

    # ── Slow detector ─────────────────────────────────────────────────────

    def _no_progress(self, x_m: float, y_m: float) -> tuple[bool, str]:
        cfg = self.config
        window = self._window

        if window.elapsed_s < cfg.stall_window_s:
            return False, ""
        if window.commanded_m < cfg.min_commanded_travel_m:
            # Not enough was asked for to out-measure the position error, so
            # the window proves nothing. Start another one.
            self._reset_window(x_m, y_m)
            return False, ""

        travelled = math.hypot(x_m - window.start_x, y_m - window.start_y)
        ratio = travelled / window.commanded_m
        elapsed = window.elapsed_s
        self._reset_window(x_m, y_m)

        if ratio < cfg.min_progress_ratio:
            # Reports the window that actually elapsed, not the configured
            # minimum. They differ whenever the fast detector has been firing,
            # because this check is skipped while it does — and quoting the
            # constant produced "asked for 208.9 m over 8 s", which is a
            # nonsense 26 m/s and made the message untrustworthy.
            return (
                True,
                f"asked for {window.commanded_m:.1f} m over "
                f"{elapsed:.0f} s but moved {travelled:.2f} m",
            )
        return False, ""

    def _reset_window(self, x_m: float, y_m: float) -> None:
        self._window = _Window(
            elapsed_s=0.0, commanded_m=0.0, start_x=x_m, start_y=y_m, started=True
        )

    # ── Main entry point ──────────────────────────────────────────────────

    def update(
        self,
        *,
        commanded_speed_mps: float,
        commanded_wheel_speeds,
        measured_wheel_speeds,
        wheel_loads,
        x_m: float,
        y_m: float,
        heading_deg: float,
        dt_s: float,
    ) -> CollisionEvent | None:
        """Fold in one control cycle. Returns an event on a NEW collision.

        Edge-triggered. The robot leans on an obstacle for many cycles while it
        backs away, and reporting each one would smear a single touch into a
        wall across the map.
        """
        if not self._window.started:
            self._reset_window(x_m, y_m)

        self._window.elapsed_s += dt_s
        self._window.commanded_m += abs(commanded_speed_mps) * dt_s

        blocked, reason, wheel = self._wheels_blocked(
            commanded_wheel_speeds, measured_wheel_speeds, wheel_loads
        )

        if blocked:
            self._agreeing_cycles += 1
        else:
            self._agreeing_cycles = 0
            self.in_contact = False

        # The slow window is evaluated every cycle, whatever the fast detector
        # is doing. Skipping it while the wheels look blocked let it run for
        # 1934 seconds instead of 8, so by the time it was consulted its verdict
        # covered most of the run and meant nothing.
        stalled, stall_reason = self._no_progress(x_m, y_m)

        fast = blocked and self._agreeing_cycles >= self.config.confirm_cycles
        detector = "wheels"
        if not fast:
            if not stalled:
                return None
            reason, detector, wheel = stall_reason, "no-progress", None

        if self.in_contact:
            # Held against something it has already reported. The robot is
            # legitimately covering no ground, so keep the window rolling or it
            # will convict a second time for the same obstacle.
            self._reset_window(x_m, y_m)
            return None

        self.in_contact = True
        # A robot held against an obstacle is legitimately not covering ground,
        # so the slow window would otherwise convict it a second time for the
        # same collision the fast detector has already reported.
        self._reset_window(x_m, y_m)
        event = CollisionEvent(
            x_m=x_m,
            y_m=y_m,
            heading_deg=heading_deg,
            reason=reason,
            detector=detector,
            wheel_index=wheel,
        )
        self.events.append(event)
        return event

    def reset(self) -> None:
        self._agreeing_cycles = 0
        self._window = _Window()
        self.in_contact = False
        self.events.clear()

    @property
    def count(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        return {
            "collisions": self.count,
            "by_wheels": sum(1 for e in self.events if e.detector == "wheels"),
            "by_no_progress": sum(
                1 for e in self.events if e.detector == "no-progress"
            ),
            "in_contact": self.in_contact,
        }
