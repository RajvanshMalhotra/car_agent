"""What temperature a particular vehicle actually runs at.

`sim/engine.py` has a single OPERATING_TEMP_C for every vehicle. That is a
guess, and it is right for at most one car: thermostat setpoints differ by tens
of degrees between a hot hatch and a truck, and corrosion is exponential in
temperature, so being wrong by 6 C is not a rounding error.

It also made the "is the game contributing" test answer differently on
different vehicles. That looked like the game being inconsistent, and was
really our one constant being wrong by a different amount each time -- which is
itself the argument for measuring this per vehicle rather than assuming it.

The setpoint is read off the run's own coolant trace: once a car is warm, a
thermostat holds it near a fixed temperature, so the settled part of the trace
*is* the answer. What has to be avoided is letting the cold start drag it down
and a hill climb drag it up.
"""

from __future__ import annotations

#: Below this a car is still warming up and says nothing about its setpoint.
#: Well under any plausible thermostat, so no vehicle is excluded by it.
NOT_WARM_ENOUGH = 70.0

#: How much of a run must be warm before the answer is trusted. A short run
#: from cold never reaches the thermostat, and the highest temperature it
#: managed is just where it stopped, not where it settles.
MIN_WARM_SAMPLES = 60

#: How the top of the warm trace is found. Not the maximum -- a single hill
#: climb or a stuck-throttle moment would become the answer -- but high enough
#: that the plateau is above it only rarely.
CEILING_QUANTILE = 0.9

#: How far below that ceiling still counts as settled. Wide enough to keep the
#: whole plateau including its normal wander under load, narrow enough to
#: exclude the tail of the warmup climb.
SETTLED_BAND_C = 3.0


def warmed_up_from(coolant_c: list[float]) -> list[float]:
    """The part of a coolant trace where the car was up to temperature."""
    return [value for value in coolant_c if value > NOT_WARM_ENOUGH]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fit_operating_temp_c(coolant_c: list[float]) -> float:
    """The temperature this vehicle settles at, from its own coolant trace.

    Neither the mean nor the maximum will do. The mean is dragged down by
    however long the warmup was, so the answer would depend on run length; the
    maximum is whatever hill the car last climbed.

    So the plateau is isolated first -- a high quantile gives its level without
    being moved by a single spike, and everything within a few degrees of that
    is the settled part -- and the answer is the middle of it. What is left out
    is the tail of the warmup climb, which is warm but still rising.

    Raises if the run never warmed up. The honest answer there is that this run
    cannot say, not a number derived from where it happened to stop.
    """
    if not coolant_c:
        raise ValueError("no coolant data to fit an operating temperature to")
    warm = warmed_up_from(coolant_c)
    if len(warm) < MIN_WARM_SAMPLES:
        raise ValueError(
            f"the car never got warm: {len(warm)} samples above "
            f"{NOT_WARM_ENOUGH:.0f} C, need {MIN_WARM_SAMPLES}. Drive for "
            f"longer before fitting an operating temperature."
        )
    ceiling = _quantile(warm, CEILING_QUANTILE)
    settled = [value for value in warm if value >= ceiling - SETTLED_BAND_C]
    return _quantile(settled, 0.5)
