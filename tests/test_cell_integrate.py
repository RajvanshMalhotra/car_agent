import pytest

from cell.aging import AgingRates, soh
from cell.integrate import CellState, run_soak, run_trip
from load.spec import BatteryScenario

RATES = AgingRates()
CRUISE = BatteryScenario(name="cruise", ambient_c=25.0)
IDLE_HOT = BatteryScenario(name="idle-hot", ambient_c=42.0, hvac=1.0, lights=True)


def _drive(seconds, speed=20.0, rpm=2500.0, coolant=90.0, running=1.0):
    for t in range(seconds):
        yield {
            "t_s": float(t), "speed_mps": speed, "rpm": rpm,
            "coolant_c": coolant, "engine_load": 0.3, "engine_running": running,
            "ax_mps2": 0.0, "ay_mps2": 0.0, "az_mps2": 9.81,
        }


def test_cruising_recharges_the_battery():
    state = CellState(soc=0.95)
    result = run_trip(_drive(600), CRUISE, state, RATES)
    assert result.state.soc > 0.95


def test_state_of_charge_is_bounded():
    state = CellState(soc=0.999)
    result = run_trip(_drive(3600), CRUISE, state, RATES)
    assert 0.0 <= result.state.soc <= 1.0


def test_idling_hot_with_everything_on_discharges_the_battery():
    # The recharge-deficit pathway. It exists only because the alternator is
    # rpm-limited and acceptance is SoC-limited.
    state = CellState(soc=1.0)
    result = run_trip(_drive(900, speed=0.0, rpm=800.0), IDLE_HOT, state, RATES)
    assert result.state.soc < 1.0
    assert any(step.i_bat_a > 0.0 for step in result.steps)


def test_a_crank_is_the_largest_current_in_the_trip():
    state = CellState(soc=1.0)
    result = run_trip(_drive(60), CRUISE, state, RATES, cranked=True)
    assert max(step.i_bat_a for step in result.steps) == pytest.approx(
        CRUISE.crank_a
    )
    assert result.steps[0].i_bat_a == pytest.approx(CRUISE.crank_a)


def test_a_crank_sags_the_terminal_voltage():
    state = CellState(soc=1.0)
    result = run_trip(_drive(60), CRUISE, state, RATES, cranked=True)
    assert result.steps[0].v_bat_v < 11.5


def test_a_hot_ambient_ages_the_battery_faster_than_a_cool_one():
    # The headline claim the whole project rests on, at its smallest scale.
    cool = run_trip(_drive(1800), BatteryScenario(name="c", ambient_c=25.0),
                    CellState(), RATES)
    hot = run_trip(_drive(1800), BatteryScenario(name="h", ambient_c=42.0),
                   CellState(), RATES)
    assert hot.damage.corrosion_equivalent_h > cool.damage.corrosion_equivalent_h


def test_the_battery_never_overshoots_the_bay_it_sits_in():
    # At the currents an SLI battery sees, self-heating is watts and
    # h (T - T_bay) is the whole thermal model, so the battery only ever
    # chases the bay.
    result = run_trip(_drive(3600), CRUISE, CellState(temp_c=25.0), RATES)
    for step in result.steps:
        assert step.t_bat_c <= step.t_bay_c + 0.5


def test_the_battery_lags_the_bay_by_hours():
    # C_th / h is 15000 / 2 = 7500 s, so half an hour into a drive the battery
    # is still well behind the bay. This is why a short trip heats the bay
    # without much heating the battery, and why soak time matters so much.
    result = run_trip(_drive(1800), CRUISE, CellState(temp_c=25.0), RATES)
    last = result.steps[-1]
    assert last.t_bat_c < last.t_bay_c - 5.0


def test_health_does_not_move_within_a_single_trip():
    # Health is held fixed for the trip, so resistance moves only with
    # temperature and state of charge -- not with age.
    state = CellState()
    result = run_trip(_drive(1800), CRUISE, state, RATES)
    assert result.steps[0].r_int_ohm == pytest.approx(
        result.steps[-1].r_int_ohm, rel=0.05
    )


def test_the_trip_still_records_the_damage_it_did():
    state = CellState()
    before = soh(state.aging, RATES)
    result = run_trip(_drive(1800), CRUISE, state, RATES)
    assert result.damage.corrosion_equivalent_h > 0.0
    assert soh(result.state.aging, RATES) < before


def test_a_soak_draws_only_the_parasitic_load():
    state = CellState(soc=1.0, temp_c=80.0)
    result = run_soak(8 * 3600, CRUISE, state, RATES)
    assert all(
        step.i_bat_a == pytest.approx(CRUISE.parasitic_a) for step in result.steps
    )
    assert result.state.soc < 1.0


def test_a_soak_still_corrodes_because_the_bay_is_still_hot():
    state = CellState(soc=1.0, temp_c=90.0)
    result = run_soak(3600, CRUISE, state, RATES)
    assert result.damage.corrosion_equivalent_h > 0.0


def test_every_step_is_reported_once_per_interval():
    result = run_trip(_drive(600), CRUISE, CellState(), RATES, dt_s=1.0)
    assert len(result.steps) == 600
