"""The fair version of "is the game contributing".

The first version compared the game's measured coolant against our engine
model's single hardcoded OPERATING_TEMP_C of 90 C. Different vehicles run at
different temperatures, so that comparison mostly measured how wrong 90 C
happened to be for whichever car was driven -- and it answered differently on
different vehicles for that reason alone.

The fair test gives the model the right steady state for *this* vehicle, fitted
from the run's own coolant trace, and asks whether the game still adds
anything. That is a much harder test to pass, and the only one worth reporting.
"""

import math

import pytest

from analysis.game_value import Sample, compare_thermal_sources
from sim.engine import OPERATING_TEMP_C


def a_run(peak_c, seconds=900, ambient=20.0, speed=18.0, throttle=0.35):
    """A steady drive whose coolant climbs from cold and holds at `peak_c`."""
    return [
        Sample(
            t_s=float(t),
            speed_mps=speed,
            rpm=2200.0,
            coolant_c=min(peak_c,
                          ambient + (peak_c - ambient) * 1.15
                          * (1 - math.exp(-t / 100.0))),
            throttle=throttle,
        )
        for t in range(seconds)
    ]


def test_the_vehicles_own_temperature_is_reported():
    result = compare_thermal_sources(a_run(97.0), ambient_c=20.0, calibrate=True)
    assert result.operating_temp_c == pytest.approx(97.0, abs=1.0)


def test_an_uncalibrated_comparison_says_it_used_the_default():
    result = compare_thermal_sources(a_run(97.0), ambient_c=20.0)
    assert result.operating_temp_c == pytest.approx(OPERATING_TEMP_C)


def test_calibrating_removes_the_disagreement_between_two_vehicles():
    # This is the actual complaint: one car said the game was contributing and
    # another said it was not. If that was our constant rather than the game,
    # calibrating each to its own temperature should bring the two verdicts
    # into line.
    hot = compare_thermal_sources(a_run(103.0), ambient_c=20.0, calibrate=True)
    cool = compare_thermal_sources(a_run(88.0), ambient_c=20.0, calibrate=True)
    assert abs(hot.ratio - cool.ratio) < 0.15


def test_the_uncalibrated_verdicts_are_the_ones_that_diverge():
    hot = compare_thermal_sources(a_run(103.0), ambient_c=20.0)
    cool = compare_thermal_sources(a_run(88.0), ambient_c=20.0)
    assert abs(hot.ratio - cool.ratio) > 0.15


def test_a_run_too_short_to_calibrate_says_so_rather_than_guessing():
    with pytest.raises(ValueError, match="warm"):
        compare_thermal_sources(a_run(97.0, seconds=60), ambient_c=20.0,
                                calibrate=True)


def test_a_calibrated_model_still_sees_only_speed_and_throttle():
    # The fitted setpoint is one number. If the model were handed the coolant
    # trace itself the test would be circular and would always agree.
    run = a_run(97.0)
    spiked = [
        Sample(s.t_s, s.speed_mps, s.rpm,
               s.coolant_c + (25.0 if 400 < s.t_s < 420 else 0.0), s.throttle)
        for s in run
    ]
    plain = compare_thermal_sources(run, ambient_c=20.0, calibrate=True)
    bumped = compare_thermal_sources(spiked, ambient_c=20.0, calibrate=True)
    assert bumped.modelled.equivalent_hours == pytest.approx(
        plain.modelled.equivalent_hours, rel=1e-6
    )
