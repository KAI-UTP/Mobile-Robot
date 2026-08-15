# RoomScan — the product view

What this is when you stop looking at it as a research project.

---

## The job it does

> *"How many square metres is this room?"*

Today that is answered by a person on their knees with a tape measure,
writing numbers on a phone, and multiplying them later. It takes ten minutes a
room, it goes wrong in rooms that are not rectangles, and the number ends up in
a quote that someone is held to.

RoomScan drives a robot around the room instead and produces a floor plan with
a measured area, saved and exportable.

## Who has that job

| | Why the area matters to them | Cost of getting it wrong |
|---|---|---|
| **Flooring installers** | material is quoted per m² | order short, second delivery kills the margin |
| **Cleaning contractors** | contracts priced per m² per visit | under-quote once, lose money every visit for a year |
| **Facilities managers** | space utilisation, compliance | reporting on numbers nobody has re-measured in years |
| **Estate agents** | listed floor area is a legal claim in some markets | misdescription |

The common thread: the number is *commercially binding*, and measuring by hand
is slow enough that people guess for the awkward rooms — which are exactly the
ones a tape measure gets wrong.

## What the MVP covers

The smallest thing that does the job end to end:

1. **Scan** — drive the robot one lap of the room
2. **Measure** — floor area, dimensions, perimeter, to about 1 %
3. **Judge** — say whether the scan is trustworthy, and why not if it isn't
4. **Save** — it is still there tomorrow, and after a rebuild
5. **Export** — a dimensioned floor plan, and a spreadsheet to quote from

That loop is complete and tested. It is what makes it a product rather than a
demonstration: a demonstration ends at step 2.

## What it deliberately does not do yet

Being explicit, because a product's boundaries matter more than its features:

- **No autonomy beyond wall-following.** It will not find its own way through a
  doorway into the next room. One room per scan.
- **No multi-room floor plan.** Rooms are measured individually and totalled;
  they are not assembled into a building layout.
- **No user accounts or cloud sync.** Scans live on the machine that ran them.
  Fine for one operator, not for a team.
- **No mobile app.** The UI is a web page; usable on a phone browser, not
  packaged.
- **Tethered.** The robot is on a USB cable to the PC, so its range is the
  cable. See the options in [BLUETOOTH-AND-GPS.md](BLUETOOTH-AND-GPS.md).
- **No IMU.** Heading comes from three omni wheels that slip sideways by
  design. This is the biggest accuracy risk on real hardware.

## Why the quality grade exists

A measuring tool that is sometimes wrong and never says so is worse than no
tool, because someone will quote from it.

Every scan is graded before it is saved:

| Grade | Meaning |
|---|---|
| **GOOD** | boundary closed, ≥85 % of the floor directly observed, pose confident |
| **ACCEPTABLE** | closed, ≥60 % observed — usable, with a wider margin |
| **POOR** | closed but thin evidence; re-scan |
| **UNUSABLE** | the boundary never closed, so the area is a **lower bound**, not a measurement |

An unclosed boundary is unusable no matter how good everything else looks —
the robot did not get all the way round, so it cannot know what it missed.

Two consequences that matter commercially:

- The **CSV total only sums usable scans**. Adding up figures already flagged
  as unreliable would produce a confident-looking total made of bad parts.
- The **exported floor plan carries the warning on the drawing itself**. A
  printed plan outlives the app, and without it a bad scan becomes a piece of
  paper that looks authoritative.

## The measurement claim

Against a known 6.00 × 4.50 m room (27.00 m²), running the full stack in
Docker:

| | Truth | Measured | Error |
|---|---|---|---|
| Floor area | 27.00 m² | 26.77 m² | **−0.9 %** |
| Long side | 6.00 m | 5.95 m | −0.8 % |
| Short side | 4.50 m | 4.50 m | 0.0 % |
| Shape overlap (IoU) | 1.000 | **0.993** | — |

For context, ±1 % on a 27 m² room is ±0.27 m² — about a quarter of a box of
laminate. Hand measurement of a non-rectangular room is routinely worse.

**This is simulated.** The figure on real hardware is unknown until the servo
bus is wired and the robot drives a real room; the simulator models wheel slip,
sonar dropout and pose drift, but it is not the same as a carpet.

## Honest positioning

Comparable products exist — laser distance meters (~RM 200) measure a wall in
seconds, and LiDAR phone apps produce rough plans. What this does that they do
not:

- **A laser meter measures walls; it does not measure a room.** Someone still
  has to decide which walls, note them down, and do the arithmetic — and
  L-shaped rooms are where that goes wrong.
- **It is unattended.** Set it going and it produces a saved, graded,
  exportable result without anyone crawling around.

Where it is weaker: slower than a laser meter for a simple rectangle, needs a
clear floor, and costs more than a tape measure.

The defensible niche is **many rooms, awkward shapes, and a number that has to
be defensible later** — which is exactly the flooring and cleaning-contract
case.

## What would take it from MVP to sellable

In the order that removes the most risk:

1. **Fit an IMU** (~RM 150). Biggest accuracy risk on real hardware.
2. **Cut the tether** — Bluetooth to the servo bus, or a mini PC onboard.
3. **Measure a real room** against a tape measure, ten times, and publish the
   spread. Every claim above is simulated until this exists.
4. **Doorway detection** so one run covers a flat rather than a room.
5. **PDF export** — what actually gets attached to a quote.
