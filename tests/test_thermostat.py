"""Finding a vehicle's operating temperature from its own run.

`sim/engine.py` holds one OPERATING_TEMP_C for every vehicle. Real cars differ
-- thermostat setpoints vary by tens of degrees -- so that constant is right
for at most one of them, and the corrosion figure is exponential in
temperature, so being wrong by 6 C is not a rounding error.

It also made `does_the_game_matter.py` answer differently on different
vehicles, which looked like the game being inconsistent and was really our own
constant being wrong by a different amount each time.
"""

import pytest

from sim.thermostat import (
    NOT_WARM_ENOUGH,
    fit_operating_temp_c,
    warmed_up_from,
)


def warming(peak, seconds=600, ambient=20.0, tau=100.0):
    """A coolant trace warming from cold and then held at `peak`.

    The hold matters. A thermostat is a valve: it stays shut until the setpoint
    and then opens, so the temperature climbs and *clamps*. An exponential
    approaching `peak` asymptotically is a different shape -- it never gets
    there, and it is still creeping at the end of the run, which is exactly
    what a real plateau is not.
    """
    import math

    return [min(peak, ambient + (peak - ambient) * 1.15 * (1 - math.exp(-t / tau)))
            for t in range(seconds)]


def test_the_settled_temperature_is_recovered_from_a_warmup():
    assert fit_operating_temp_c(warming(96.0)) == pytest.approx(96.0, abs=0.5)


def test_a_different_vehicle_gives_a_different_answer():
    hot = fit_operating_temp_c(warming(103.0))
    cool = fit_operating_temp_c(warming(88.0))
    assert hot > cool + 10.0


def test_the_cold_start_is_not_averaged_into_the_answer():
    # Naively averaging the whole trace drags the answer far below the
    # setpoint, by an amount that depends on how long the run was.
    trace = warming(96.0, seconds=600)
    assert fit_operating_temp_c(trace) > sum(trace) / len(trace) + 5.0


def test_a_run_that_never_warmed_up_is_refused_rather_than_guessed():
    # Two minutes from cold never reaches the thermostat, and reporting the
    # highest temperature seen would just report where the run stopped.
    with pytest.raises(ValueError, match="warm"):
        fit_operating_temp_c(warming(96.0, seconds=90))


def test_an_empty_trace_is_refused():
    with pytest.raises(ValueError):
        fit_operating_temp_c([])


def test_only_the_warmed_part_of_a_run_is_used():
    trace = warming(96.0, seconds=600)
    warm = warmed_up_from(trace)
    assert all(value > NOT_WARM_ENOUGH for value in warm)
    assert len(warm) < len(trace)


def test_a_trace_that_is_warm_throughout_is_used_whole():
    trace = [95.0] * 300
    assert len(warmed_up_from(trace)) == 300


def test_noise_does_not_move_the_answer_much():
    import random

    rng = random.Random(0)
    clean = warming(96.0)
    noisy = [value + rng.uniform(-1.5, 1.5) for value in clean]
    assert fit_operating_temp_c(noisy) == pytest.approx(
        fit_operating_temp_c(clean), abs=1.0
    )


def test_a_brief_overheat_does_not_become_the_setpoint():
    # A long climb spikes the coolant above the thermostat. Taking the maximum
    # would call that spike the operating temperature for the whole run.
    trace = warming(96.0)
    trace[400] = 118.0
    assert fit_operating_temp_c(trace) < 100.0
