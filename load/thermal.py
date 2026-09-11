"""Under-bonnet temperature: the number the dominant ageing pathway runs on.

Nothing measures it. BeamNG models coolant and oil temperature and OutGauge
reports coolant, but neither is the temperature of the air around the battery,
and BeamNG does not model an engine bay at all. So it is estimated here, and
the same estimator runs unchanged on real vehicle telemetry.

Measured coolant is used verbatim. The model never integrates over a
measurement -- it only ever lags behind one.

The structure is defensible; the coefficients are not calibrated against a real
engine bay. Ratios between conditions are the usable output and absolute
temperatures are indicative, which is a caveat that must travel with every
number this module produces.

**Heat soak.** `sim/engine.py` set the coupling to zero the moment the engine
stopped, which says the bay falls to ambient immediately. It does the opposite.
Airflow stops while the block is still at 130 C, so the bay climbs for several
minutes before decaying with the coolant -- and that peak is both the hottest
the battery ever gets and, corrosion being Arrhenius, disproportionately
damaging. `BAY_COUPLING_SOAK` is higher than `BAY_COUPLING_STATIC` for exactly
this reason.
"""

from __future__ import annotations

#: Fraction of the coolant-to-ambient rise the bay sees with the engine running
#: and no airflow.
BAY_COUPLING_STATIC = 0.62

#: And with the engine stopped. Higher, not lower: no fan, no airflow, and a
#: block still at operating temperature radiating into still air.
BAY_COUPLING_SOAK = 0.95

#: Extra bay heating at full load.
BAY_LOAD_GAIN = 0.25

#: How quickly road speed carries bay heat away, per m/s.
BAY_AIRFLOW_GAIN = 0.10

#: Bay thermal lag, seconds. Air responds faster than coolant.
BAY_TAU_S = 60.0

#: Cool-down time constant of a stopped engine, seconds. Coolant sheds its heat
#: in about a quarter of an hour; the battery, with a time constant of hours,
#: does not. The two must not be conflated -- using battery temperature as a
#: stand-in for coolant during a soak keeps the bay hot for hours that never
#: happened, in a term that is exponential in temperature.
COOLDOWN_TAU_S = 900.0

READS: tuple[str, ...] = ("coolant_c", "speed_mps", "engine_load", "engine_running")


def bay_target_c(
    coolant_c: float,
    speed_mps: float,
    engine_load: float,
    engine_running: float,
    ambient_c: float,
) -> float:
    """The temperature the bay is heading towards right now.

    Never below ambient: the bay cannot be colder than the air being drawn
    through it, whatever the coolant reads.
    """
    if engine_running >= 0.5:
        coupling = (
            BAY_COUPLING_STATIC
            * (1.0 + BAY_LOAD_GAIN * min(1.0, max(0.0, engine_load)))
            / (1.0 + BAY_AIRFLOW_GAIN * max(0.0, speed_mps))
        )
    else:
        coupling = BAY_COUPLING_SOAK
    rise = coupling * (coolant_c - ambient_c)
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
            speed_mps=float(sample["speed_mps"]),
            engine_load=float(sample["engine_load"]),
            engine_running=float(sample["engine_running"]),
            ambient_c=self.ambient_c,
        )
        # Clamped so a step longer than the time constant cannot overshoot.
        alpha = min(1.0, dt_s / BAY_TAU_S)
        self.temperature_c += (target - self.temperature_c) * alpha
        return self.temperature_c
