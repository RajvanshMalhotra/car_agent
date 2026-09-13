"""Under-bonnet temperature: the number the dominant ageing pathway runs on.

Nothing measures it directly. BeamNG models coolant and oil temperature, and a
real OBD-II adapter reports both too (`e.watertemp`, `e.oiltemp` -- see
`collect/schema.py`). Neither is the temperature of the air around the
battery, and neither BeamNG nor a real car models an engine bay at all. So bay
temperature is estimated here, from the two thermal channels that ARE
measured, and the same estimator runs unchanged on real vehicle telemetry and
on BeamNG's channels.

Measured coolant and oil are used verbatim. The model never integrates over a
measurement -- it only ever lags behind one.

**What changed from the first version.** The original estimator computed a
coupling fraction from three invented constants: a static base coupling, a
load gain, and an airflow gain applied to road speed. None were calibrated
against a real engine bay -- they were guesses about how load and airflow
*should* behave.

The project's own recording carries a second measured thermal node, `oil_c`,
and the oil-minus-coolant gap turns out to be exactly the airflow signal the
old model was guessing at, except measured per sample instead of assumed as a
constant:

     0-5  m/s   +19.21 C   (stationary or crawling -- oil sits close to the
                            hot metal, coolant is thermostat-regulated, and
                            with no airflow nothing pulls the two together)
     5-10 m/s   +1.05 C
    10-15 m/s   +0.68 C
    15-20 m/s   +0.53 C
    20-25 m/s   +0.65 C
    25-30 m/s   +0.48 C   (cruising -- the gap collapses to under a degree
                           the moment air is moving through the bay)
    engine off  -1.20 C   (oil cools very slightly faster than the coolant
                           loop once the pump stops; the gap goes negative)

So the new estimator interpolates between ambient and the hotter of the two
measured nodes, with the interpolation fraction (the "coupling") set by the
measured gap rather than assumed from speed and load. A large gap means
stagnant air and poor heat removal -- the bay sits closer to the hot node. A
small gap means airflow is carrying heat away -- the bay sits closer to
ambient. Engine load is no longer a separate term: harder driving already
shows up as a larger gap and a hotter oil reading, so a load gain on top of a
channel that already encodes load would be double counting.

**The coupling is bounded, not free to approach 1.0.** The first cut of this
rework let the gap-driven coupling approach 1.0 at a large gap, which lets the
bay track the hot anchor almost one-for-one. Against this project's own
recording that produced a 195 C bay -- hot enough to soften a polypropylene
battery case (130-150 C) and destroy the battery in weeks, not the multi-year
life the model then went on to predict. That is wrong on physical grounds
regardless of how extreme the measured anchor is: bay AIR is not the metal it
surrounds. It sits in a vented compartment with convective and radiative loss
to the body shell and to ambient, and the battery is typically sited away
from the hottest components, so even fully stagnant bay air stays well below
the hot node -- real under-bonnet air peaks around 60-100 C. The measured gap
still decides WHERE in the physical range the bay sits -- that is the actual
improvement, and it stays -- but the range itself is now a bounded interval,
`COUPLING_VENTILATED` to `COUPLING_STAGNANT`, not `[0, 1)`. See their
definitions below for the arithmetic against the 60-100 C figure and against
this recording's own worst sample.

`bay_target_c` also now clamps its result to never exceed the hot anchor,
defensively, in addition to the coupling bound already keeping it there --
air surrounded by metal at temperature T cannot exceed T, and a hard clamp
means that stays true even if a future change to the coupling arithmetic
would otherwise let it drift past 1.0 again.

**This is still not a measurement of the bay.** Anchoring to oil and coolant
and gating on their gap is better than gating on invented numbers, but it
remains an inference from two other numbers, run through a function this
module chose. And BeamNG's oil temperature is itself a simulation output --
a model's guess at oil temperature -- so on the sim side this anchors one
model to another model's number, which is better than an invented coefficient
and is still not a measurement. On the real-telemetry side `oil_c` is a
genuine OBD-II reading, so the anchor is real there.

**Free parameters: four.** The first cut of this rework replaced three
invented constants (`BAY_COUPLING_STATIC`, `BAY_LOAD_GAIN`,
`BAY_AIRFLOW_GAIN`) with one (`GAP_SCALE_C`) and called that the win. It
wasn't: an unbounded two-parameter model that predicts a battery-destroying
195 C bay is worse than a four-parameter model that stays physically
plausible. The real win is that the airflow term is now measured (the
oil-coolant gap) instead of assumed (road speed through an invented gain).
What remains:

  `GAP_SCALE_C`         -- degrees of oil-coolant gap it takes for the
                            engine-running coupling to move most of the way
                            from `COUPLING_VENTILATED` to `COUPLING_STAGNANT`.
  `COUPLING_VENTILATED` -- the engine-running coupling floor, gap near zero.
  `COUPLING_STAGNANT`   -- the engine-running coupling ceiling, gap large.
                            NOT 1.0 -- see above and the constant's own
                            docstring for the arithmetic.
  `BAY_COUPLING_SOAK`   -- carried over unchanged from the first version. It
                            governs a different regime (engine off) that the
                            gap cannot drive, because the measured engine-off
                            gap is small and slightly NEGATIVE rather than
                            large and positive -- see the heat-soak note
                            below. It was already a bounded sub-1.0 constant
                            (0.95, fixed, never gap-driven), so it was never
                            the mechanism that produced the 195 C figure.

`COOLDOWN_TAU_S` and `BAY_TAU_S` are lag time constants, not coupling
fudges, and are unchanged.

Ratios between conditions remain the usable output; absolute temperatures
remain indicative.

**Heat soak.** `sim/engine.py` set the coupling to zero the moment the engine
stopped, which says the bay falls to ambient immediately. It does the
opposite. Airflow stops while the block is still at 130 C, so the bay climbs
for several minutes before decaying with the coolant -- and that peak is both
the hottest the battery ever gets and, corrosion being Arrhenius,
disproportionately damaging. The measured gap goes slightly negative during
this regime (-1.2 C), which a gap-based coupling would read as "well
ventilated" and collapse towards ambient -- backwards for a stationary engine
radiating into still air. That is exactly why engine-off is handled as its
own branch with its own constant, `BAY_COUPLING_SOAK`, rather than folded into
the gap mapping.
"""

from __future__ import annotations

import math

#: Degrees of oil-minus-coolant gap it takes for the engine-running coupling
#: to move most of the way from COUPLING_VENTILATED to COUPLING_STAGNANT. At
#: GAP_SCALE_C degrees of gap, it has covered 1 - 1/e (~63%) of that
#: distance; the measured stationary/crawling gap of +19.21 C is more than
#: three time-constants past that, i.e. deep into the stagnant end, while the
#: measured cruising gaps of well under +1 C sit close to the ventilated end.
GAP_SCALE_C = 6.0

#: Engine-running coupling floor: airflow doing everything it can (gap near
#: zero). A running engine still radiates into the bay regardless of road
#: speed, so this is not 0 -- at typical cruising coolant (~90-100 C) and
#: 25 C ambient it puts the bay around 48-53 C, near the low end of the
#: measured real-world 60-100 C under-bonnet range.
COUPLING_VENTILATED = 0.35

#: Engine-running coupling ceiling: airflow doing nothing (gap large).
#: Deliberately NOT 1.0 -- bay air is not the metal it surrounds, and even
#: fully stagnant it stays well below the hot node (see the module
#: docstring). Chosen against this project's own worst measured sample: a
#: stationary high-RPM event drives oil to 212.5 C while coolant sits pinned
#: at its 130 C cap. At 0.45 that anchor produces a bay of about 110 C --
#: hot, clearly hotter than the same moment showed up in the old
#: coolant-only model (90.6 C), but short of a polypropylene battery case's
#: 130-150 C softening point. 0.75 (a naively "stagnant should be close to
#: the anchor" guess) was tried first and rejected: against this same 212.5 C
#: sample it gives a 166 C bay, which is exactly the unphysical result this
#: bound exists to prevent.
COUPLING_STAGNANT = 0.45

#: Coupling fraction with the engine stopped. Governs a different regime
#: (engine off) that the gap cannot drive -- see the module docstring for why.
#: Higher than the engine-running ceiling: no fan, no airflow, and a block
#: still at operating temperature radiating into still air. It was already a
#: fixed sub-1.0 constant before this pass (never gap-driven), so it was
#: never the mechanism that let the bay approach the hot anchor unbounded.
BAY_COUPLING_SOAK = 0.95

#: Bay thermal lag, seconds. Air responds faster than coolant.
BAY_TAU_S = 60.0

#: Cool-down time constant of a stopped engine, seconds. Coolant sheds its heat
#: in about a quarter of an hour; the battery, with a time constant of hours,
#: does not. The two must not be conflated -- using battery temperature as a
#: stand-in for coolant during a soak keeps the bay hot for hours that never
#: happened, in a term that is exponential in temperature.
COOLDOWN_TAU_S = 900.0

READS: tuple[str, ...] = ("coolant_c", "oil_c", "engine_running")


def _gap_coupling(gap_c: float) -> float:
    """Map a measured oil-coolant gap to an engine-running coupling.

    Bounded to `[COUPLING_VENTILATED, COUPLING_STAGNANT]`, never `[0, 1)` --
    see the module docstring for why 1.0 is physically wrong regardless of
    how large the measured gap gets. Saturating in the gap within that range:
    near 0 C (well ventilated) the coupling sits at the floor, and by the
    time the gap reaches the +19-20 C that stationary/crawling conditions
    actually measure, it has covered most of the distance to the ceiling. A
    negative gap clamps to the floor here -- the regime that produces a
    negative gap (engine off) is handled by the caller with its own
    constant, never by this function.
    """
    positive_gap = max(0.0, gap_c)
    saturation = 1.0 - math.exp(-positive_gap / GAP_SCALE_C)
    return COUPLING_VENTILATED + (COUPLING_STAGNANT - COUPLING_VENTILATED) * saturation


def bay_target_c(
    coolant_c: float,
    oil_c: float,
    engine_running: float,
    ambient_c: float,
) -> float:
    """The temperature the bay is heading towards right now.

    The hot anchor is the hotter of the two measured nodes. Oil generally
    runs hottest and sits closest to the hot metal, but a cold engine can
    invert that (e.g. just after a cold start, or a soak's own synthesised
    oil trailing coolant down) -- taking the max handles both without a
    special case.

    With the engine running, the coupling comes from the measured
    oil-coolant gap (see `_gap_coupling`). With the engine off, the gap is
    small and slightly negative (see the module docstring), so the engine-off
    coupling is a fixed constant, `BAY_COUPLING_SOAK`, instead.

    Never below ambient: the bay cannot be colder than the air being drawn
    through it, whatever the coolant or oil read. Never above the hot anchor
    either: air surrounded by metal at temperature T cannot exceed T.
    `COUPLING_VENTILATED`, `COUPLING_STAGNANT` and `BAY_COUPLING_SOAK` are
    all already below 1.0, so the upper clamp below is defensive rather than
    load-bearing today -- it exists so a future change to the coupling
    arithmetic cannot silently reproduce the unbounded-coupling bug this
    function used to have.
    """
    hot_anchor_c = max(coolant_c, oil_c)
    if engine_running >= 0.5:
        coupling = _gap_coupling(oil_c - coolant_c)
    else:
        coupling = BAY_COUPLING_SOAK
    rise = coupling * (hot_anchor_c - ambient_c)
    bay_c = ambient_c + max(0.0, rise)
    return min(bay_c, hot_anchor_c) if hot_anchor_c >= ambient_c else bay_c


class BayTemperature:
    """First-order lag towards `bay_target_c`."""

    def __init__(self, ambient_c: float, initial_c: float | None = None) -> None:
        self.ambient_c = ambient_c
        self.temperature_c = ambient_c if initial_c is None else initial_c

    def step(self, sample, dt_s: float) -> float:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        target = bay_target_c(
            coolant_c=float(sample["coolant_c"]),
            oil_c=float(sample["oil_c"]),
            engine_running=float(sample["engine_running"]),
            ambient_c=self.ambient_c,
        )
        # Clamped so a step longer than the time constant cannot overshoot.
        alpha = min(1.0, dt_s / BAY_TAU_S)
        self.temperature_c += (target - self.temperature_c) * alpha
        return self.temperature_c
