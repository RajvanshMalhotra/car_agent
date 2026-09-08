"""Turning driving into battery current.

This is the interface the whole downstream argument rests on. The real dataset's
usable columns are current and voltage; the simulator produces neither, and
until driving is converted into current the simulated and measured data do not
even share a representation -- so the masked real-data latents, the one thing
breaking the circularity, have nothing to fuse with.

It also delivers two ageing pathways that were producing no data at all. The
recharge deficit *is* net discharge at idle with the alternator turning too
slowly to cover the load. Sulfation follows cranking and the state of charge
history that follows it.
"""

import pytest

from battery.electrical import (
    CRANK_CURRENT_A,
    ElectricalModel,
    alternator_output_a,
)


def a_model(**kwargs):
    return ElectricalModel(**kwargs)


# -- the alternator --------------------------------------------------------


def test_a_stopped_engine_generates_nothing():
    assert alternator_output_a(rpm=0.0, rated_a=120.0) == 0.0


def test_an_idling_engine_generates_something_but_not_much():
    idle = alternator_output_a(rpm=800.0, rated_a=120.0)
    assert 0.0 < idle < 120.0 * 0.6


def test_output_rises_with_engine_speed():
    assert (alternator_output_a(rpm=2500.0, rated_a=120.0)
            > alternator_output_a(rpm=900.0, rated_a=120.0))


def test_output_is_capped_at_the_rating():
    assert alternator_output_a(rpm=6000.0, rated_a=120.0) == pytest.approx(120.0)


# -- accessory load --------------------------------------------------------


def test_a_car_doing_nothing_still_draws_current():
    # Engine management, ignition, fuel pump, instruments.
    assert a_model().accessory_load_a(hvac=0.0, lights=False) > 10.0


def test_climate_control_adds_load():
    model = a_model()
    assert model.accessory_load_a(hvac=1.0, lights=False) > model.accessory_load_a(
        hvac=0.0, lights=False
    )


def test_lights_add_load():
    model = a_model()
    assert model.accessory_load_a(hvac=0.0, lights=True) > model.accessory_load_a(
        hvac=0.0, lights=False
    )


# -- net battery current ---------------------------------------------------


def test_cruising_charges_the_battery():
    # Positive current is discharge, so charging is negative.
    model = a_model()
    assert model.battery_current_a(rpm=2500.0, hvac=0.5) < 0.0


def test_idling_with_everything_on_can_discharge_it():
    # The recharge-deficit pathway: the alternator at idle may not cover the
    # load, and the battery makes up the difference.
    model = a_model(alternator_rated_a=90.0)
    assert model.battery_current_a(rpm=750.0, hvac=1.0, lights=True) > 0.0


def test_idling_with_nothing_on_still_charges():
    model = a_model()
    assert model.battery_current_a(rpm=800.0, hvac=0.0, lights=False) < 0.0


def test_a_stopped_engine_draws_from_the_battery():
    model = a_model()
    assert model.battery_current_a(rpm=0.0, hvac=0.0, lights=False) > 0.0


# -- cranking --------------------------------------------------------------


def test_cranking_draws_hundreds_of_amps():
    assert 150.0 < CRANK_CURRENT_A < 700.0


def test_a_crank_shows_as_a_large_brief_discharge():
    model = a_model()
    current = model.battery_current_a(rpm=0.0, hvac=0.0, lights=False, cranking=True)
    assert current > 150.0


# -- terminal voltage ------------------------------------------------------


def test_a_charging_battery_sits_at_the_regulated_voltage():
    model = a_model()
    assert model.terminal_voltage_v(current_a=-40.0, rpm=2500.0) == pytest.approx(
        14.2, abs=0.4
    )


def test_a_resting_battery_sits_near_its_open_circuit_voltage():
    model = a_model()
    assert 12.0 < model.terminal_voltage_v(current_a=5.0, rpm=0.0) < 13.0


def test_cranking_pulls_the_voltage_down_hard():
    model = a_model()
    cranking = model.terminal_voltage_v(current_a=CRANK_CURRENT_A, rpm=0.0)
    assert 7.0 < cranking < 11.0


def test_a_colder_battery_sags_more_when_cranked():
    warm = ElectricalModel(temperature_c=25.0)
    cold = ElectricalModel(temperature_c=-10.0)
    assert cold.terminal_voltage_v(CRANK_CURRENT_A, rpm=0.0) < warm.terminal_voltage_v(
        CRANK_CURRENT_A, rpm=0.0
    )


def test_a_worn_battery_sags_more_than_a_healthy_one():
    healthy = ElectricalModel(internal_resistance_ohm=0.006)
    worn = ElectricalModel(internal_resistance_ohm=0.020)
    assert worn.terminal_voltage_v(200.0, rpm=0.0) < healthy.terminal_voltage_v(
        200.0, rpm=0.0
    )


# -- the two channels the real data has ------------------------------------


def test_a_driving_state_becomes_current_and_voltage():
    model = a_model()
    current, voltage = model.from_driving(
        rpm=2200.0, engine_on=True, hvac=0.5, lights=False, cranking=False
    )
    assert current < 0.0
    assert 13.0 < voltage < 15.0


def test_an_idle_state_with_load_becomes_a_discharge():
    model = a_model(alternator_rated_a=85.0)
    current, voltage = model.from_driving(
        rpm=700.0, engine_on=True, hvac=1.0, lights=True, cranking=False
    )
    assert current > 0.0
    assert voltage < 13.5
