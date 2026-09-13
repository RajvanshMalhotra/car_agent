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
stagnant air and poor heat removal -- the bay sits close to the hot node. A
small gap means airflow is carrying heat away -- the bay sits close to
ambient. Engine load is no longer a separate term: harder driving already
shows up as a larger gap and a hotter oil reading, so a load gain on top of a
channel that already encodes load would be double counting.

**This is still not a measurement of the bay.** Anchoring to oil and coolant
and gating on their gap is better than gating on three invented numbers, but
it remains an inference from two other numbers, run through a function this
module chose. And BeamNG's oil temperature is itself a simulation output --
a model's guess at oil temperature -- so on the sim side this anchors one
model to another model's number, which is better than an invented coefficient
and is still not a measurement. On the real-telemetry side `oil_c` is a
genuine OBD-II reading, so the anchor is real there.

**Free parameters: two, down from three.** `BAY_COUPLING_STATIC`,
`BAY_LOAD_GAIN` and `BAY_AIRFLOW_GAIN` are gone. What remains:

  `GAP_SCALE_C`       -- degrees of oil-coolant gap it takes for the
                          engine-running coupling to approach the stagnant
                          (hot-anchor) end. The single scale parameter on the
                          gap-to-coupling mapping asked for by the rework.
  `BAY_COUPLING_SOAK` -- carried over unchanged from the first version. It is
                          not one of the three being replaced: it governs a
                          different regime (engine off) that the gap cannot
                          drive, because the measured engine-off gap is small
                          and slightly NEGATIVE rather than large and
                          positive -- see the heat-soak note below.

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
#: to cover most of the distance from ambient to the hot anchor. The single
#: free scale parameter this rework introduces, replacing the three invented
#: constants above. At GAP_SCALE_C degrees of gap, coupling has covered
#: 1 - 1/e (~63%) of its range; the measured stationary/crawling gap of
#: +19.21 C is more than three time-constants past that, i.e. deep into the
#: stagnant end, while the measured cruising gaps of well under +1 C sit
#: close to the well-ventilated end.
GAP_SCALE_C = 6.0

#: Coupling fraction with the engine stopped. Not one of the three constants
#: this rework replaces -- see the module docstring for why the gap cannot
#: drive this regime. Higher than a typical driving coupling: no fan, no
#: airflow, and a block still at operating temperature radiating into still
#: air.
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
    """Map a measured oil-coolant gap to an engine-running coupling in [0, 1).

    Saturating in the gap: near 0 C (well ventilated) the coupling is near 0,
    and by the time the gap reaches the +19-20 C that stationary/crawling
    conditions actually measure, the coupling has covered most of its range
    towards 1 (stagnant, bay close to the hot anchor). A negative gap clamps
    to 0 here -- the regime that produces a negative gap (engine off) is
    handled by the caller with its own constant, never by this function.
    """
    positive_gap = max(0.0, gap_c)
    return 1.0 - math.exp(-positive_gap / GAP_SCALE_C)


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
    through it, whatever the coolant or oil read.
    """
    hot_anchor_c = max(coolant_c, oil_c)
    if engine_running >= 0.5:
        coupling = _gap_coupling(oil_c - coolant_c)
    else:
        coupling = BAY_COUPLING_SOAK
    rise = coupling * (hot_anchor_c - ambient_c)
    return ambient_c + max(0.0, rise)


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
