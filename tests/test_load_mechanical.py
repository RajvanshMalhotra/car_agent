import math

import pytest

from load.mechanical import READS, road_load_n, mech_power_w
from load.spec import BatteryScenario

SCENARIO = BatteryScenario(name="t", mass_kg=1500.0, cd_a_m2=0.75, crr=0.012,
                           air_density=1.2)


def _sample(speed=0.0, a_long=0.0, grade=0.0, mass=1500.0):
    return {"speed_mps": speed, "a_long_mps2": a_long, "grade_rad": grade,
            "mass_kg": mass}


def test_a_stationary_car_on_the_flat_needs_only_rolling_resistance():
    force = road_load_n(_sample(), SCENARIO)
    assert force == pytest.approx(0.012 * 1500.0 * 9.81, rel=1e-6)


def test_grade_adds_its_own_weight_component():
    flat = road_load_n(_sample(), SCENARIO)
    uphill = road_load_n(_sample(grade=0.05), SCENARIO)
    assert uphill - flat == pytest.approx(
        1500.0 * 9.81 * math.sin(0.05) - 0.012 * 1500.0 * 9.81 * (1 - math.cos(0.05)),
        rel=1e-6,
    )


def test_drag_grows_with_the_square_of_speed():
    at_10 = road_load_n(_sample(speed=10.0), SCENARIO)
    at_20 = road_load_n(_sample(speed=20.0), SCENARIO)
    rolling = 0.012 * 1500.0 * 9.81
    assert (at_20 - rolling) == pytest.approx(4.0 * (at_10 - rolling), rel=1e-6)


def test_acceleration_uses_the_measured_mass_not_the_scenario_default():
    # mass_kg is a trajectory column; a heavier recorded car must show up here.
    light = road_load_n(_sample(a_long=2.0, mass=1000.0), SCENARIO)
    heavy = road_load_n(_sample(a_long=2.0, mass=2000.0), SCENARIO)
    assert heavy > light


def test_power_is_force_times_speed():
    sample = _sample(speed=20.0, a_long=1.0)
    assert mech_power_w(sample, SCENARIO) == pytest.approx(
        road_load_n(sample, SCENARIO) * 20.0, rel=1e-9
    )


def test_a_stationary_car_does_no_mechanical_work():
    assert mech_power_w(_sample(speed=0.0), SCENARIO) == pytest.approx(0.0)


def test_reads_declares_every_column_it_touches():
    assert set(READS) == {"speed_mps", "a_long_mps2", "grade_rad", "mass_kg"}
