import pytest

from cell.ecm import open_circuit_v, r_int_ohm, terminal_v


def test_a_full_battery_rests_at_about_12_7_volts():
    assert open_circuit_v(1.0) == pytest.approx(12.7)


def test_open_circuit_voltage_falls_with_state_of_charge():
    assert open_circuit_v(0.5) < open_circuit_v(1.0)


def test_state_of_charge_outside_the_range_is_clamped_not_extrapolated():
    assert open_circuit_v(1.4) == pytest.approx(open_circuit_v(1.0))
    assert open_circuit_v(-0.2) == pytest.approx(open_circuit_v(0.0))


def test_a_healthy_warm_full_battery_is_at_its_nominal_resistance():
    assert r_int_ohm(0.006, soh=1.0, temp_c=25.0, soc=1.0) == pytest.approx(0.006)


def test_resistance_rises_as_the_battery_ages():
    assert r_int_ohm(0.006, 0.8, 25.0, 1.0) > r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_resistance_rises_in_the_cold():
    # Why a marginal battery survives all summer and fails on the first frost.
    assert r_int_ohm(0.006, 1.0, -18.0, 1.0) > 1.5 * r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_resistance_rises_as_the_battery_discharges():
    assert r_int_ohm(0.006, 1.0, 25.0, 0.5) > r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_a_discharging_battery_sags_below_its_open_circuit_voltage():
    v = terminal_v(soc=1.0, current_a=350.0, r_int_ohm=0.006,
                   alternator_capability_a=0.0)
    assert v == pytest.approx(12.7 - 350.0 * 0.006)


def test_a_worn_battery_sags_further_on_the_same_crank():
    healthy = terminal_v(1.0, 350.0, r_int_ohm(0.006, 1.0, 25.0, 1.0), 0.0)
    worn = terminal_v(1.0, 350.0, r_int_ohm(0.006, 0.8, 25.0, 1.0), 0.0)
    assert worn < healthy


def test_the_regulator_holds_the_bus_up_while_the_alternator_is_working():
    # Charging, the reading says more about the alternator than the battery.
    v = terminal_v(soc=0.9, current_a=-20.0, r_int_ohm=0.006,
                   alternator_capability_a=120.0)
    assert 13.8 < v < 14.3


def test_a_charging_current_with_no_alternator_is_not_a_regulated_bus():
    v = terminal_v(soc=0.9, current_a=-20.0, r_int_ohm=0.006,
                   alternator_capability_a=0.0)
    assert v < 13.0
