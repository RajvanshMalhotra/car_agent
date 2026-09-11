import pytest

from load.electrical import (
    CUT_IN_SHAFT_RPM,
    accessory_load_a,
    acceptance_temperature_factor,
    alternator_capability_a,
    battery_current_a,
    charge_acceptance_a,
)
from load.spec import BatteryScenario

IDLE = BatteryScenario(name="idle", alternator_rated_a=120.0, accessory_base_a=22.0)
LOADED = BatteryScenario(name="loaded", alternator_rated_a=120.0,
                         accessory_base_a=22.0, lights=True, hvac=1.0)


def test_the_alternator_produces_nothing_below_cut_in():
    rpm = (CUT_IN_SHAFT_RPM / 2.6) - 100.0
    assert alternator_capability_a(rpm, 120.0) == pytest.approx(0.0)


def test_the_alternator_saturates_at_its_rating():
    assert alternator_capability_a(6000.0, 120.0) == pytest.approx(120.0)


def test_the_alternator_gives_roughly_a_quarter_of_its_rating_at_idle():
    # 800 rpm through a 2.6 pulley is a 2080 rpm shaft, a quarter of the way
    # from cut-in to full output.
    assert alternator_capability_a(800.0, 120.0) == pytest.approx(27.8, abs=1.0)


def test_accessory_load_adds_lights_and_blower():
    assert accessory_load_a(IDLE) == pytest.approx(22.0)
    assert accessory_load_a(LOADED) == pytest.approx(22.0 + 12.0 + 28.0)


def test_a_full_battery_accepts_no_charge():
    assert charge_acceptance_a(soc=1.0, temp_c=25.0, scenario=IDLE) == pytest.approx(0.0)


def test_acceptance_grows_as_the_battery_empties():
    nearly_full = charge_acceptance_a(0.99, 25.0, IDLE)
    depleted = charge_acceptance_a(0.90, 25.0, IDLE)
    assert nearly_full == pytest.approx(1.2, abs=0.1)
    assert depleted == pytest.approx(12.0, abs=0.5)


def test_a_cold_battery_accepts_charge_poorly():
    assert acceptance_temperature_factor(25.0) == pytest.approx(1.0)
    assert acceptance_temperature_factor(-10.0) < 0.5
    assert charge_acceptance_a(0.9, -10.0, IDLE) < charge_acceptance_a(0.9, 25.0, IDLE)


def test_cranking_is_a_large_discharge_whatever_else_is_true():
    current = battery_current_a(rpm=0.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=0.0, cranking=True)
    assert current == pytest.approx(IDLE.crank_a)


def test_a_parked_car_draws_its_parasitic_load():
    current = battery_current_a(rpm=0.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=0.0)
    assert current == pytest.approx(IDLE.parasitic_a)


def test_idling_with_everything_on_runs_a_deficit():
    # 62 A of load against about 28 A of alternator. This is the recharge
    # deficit pathway, and it only exists because the alternator is rpm-limited.
    current = battery_current_a(rpm=800.0, soc=1.0, temp_c=25.0, scenario=LOADED,
                                engine_running=1.0)
    assert current > 30.0


def test_cruising_charges_the_battery():
    current = battery_current_a(rpm=2500.0, soc=0.95, temp_c=25.0, scenario=IDLE,
                                engine_running=1.0)
    assert current < 0.0


def test_a_full_battery_at_cruise_draws_only_what_the_accessories_need():
    # With no acceptance headroom the alternator supplies the load and no more,
    # so the battery current is essentially zero rather than a phantom charge.
    current = battery_current_a(rpm=2500.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=1.0)
    assert current == pytest.approx(0.0, abs=0.01)
