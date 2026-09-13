import pytest

from cell.aging import AgingRates, soh
from cell.integrate import CellState, run_soak, run_trip
from load.spec import BatteryScenario
from load.thermal import bay_target_c

RATES = AgingRates()
CRUISE = BatteryScenario(name="cruise", ambient_c=25.0)
IDLE_HOT = BatteryScenario(name="idle-hot", ambient_c=42.0, hvac=1.0, lights=True)


#: Measured oil-minus-coolant gap while stationary/crawling (see
#: `load/thermal.py`'s docstring). `_drive(speed=0.0, ...)` is used for the
#: idling/stop-go scenarios below, where this is the realistic gap.
STATIONARY_GAP_C = 19.21

#: Measured gap around highway speed. `_drive`'s default speed (20 m/s) sits
#: in the 15-20 m/s bin.
CRUISING_GAP_C = 0.53


def _drive(seconds, speed=20.0, rpm=2500.0, coolant=90.0, running=1.0,
           oil_gap=None):
    gap = (STATIONARY_GAP_C if speed <= 0.0 else CRUISING_GAP_C) \
        if oil_gap is None else oil_gap
    for t in range(seconds):
        yield {
            "t_s": float(t), "speed_mps": speed, "rpm": rpm,
            "coolant_c": coolant, "oil_c": coolant + gap,
            "engine_load": 0.3, "engine_running": running,
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
    # Stationary/idling (speed=0) is used so the bay is meaningfully hot -- a
    # sustained highway gap collapses to under a degree (see
    # `load/thermal.py`), which would leave both temperatures near ambient
    # and the lag too small to demonstrate anything.
    result = run_trip(_drive(1800, speed=0.0), CRUISE, CellState(temp_c=25.0),
                      RATES)
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


def test_successive_trips_return_distinct_state_snapshots():
    # TripResult.state used to be the same mutable CellState object every
    # call, so every stored result's aging silently tracked whatever the
    # caller's live state became later. Each result must freeze its own.
    state = CellState()
    first = run_trip(_drive(600), CRUISE, state, RATES)
    second = run_trip(_drive(600), CRUISE, state, RATES)
    assert first.state is not second.state
    assert first.state.aging is not second.state.aging
    assert soh(first.state.aging, RATES) != soh(second.state.aging, RATES)
    # And the first snapshot must not have silently become the second's
    # values just because the caller's `state` object kept accumulating.
    assert soh(first.state.aging, RATES) > soh(second.state.aging, RATES)


def test_an_explicit_health_overrides_the_derived_one():
    # cell/life.py's resim-on-drift loop hands the probed trip a health value
    # that has already drifted past the caller's own aging state. Without an
    # override, run_trip silently re-derives health from state.aging instead,
    # and the whole two-rate feedback loop goes inert.
    healthy = run_trip(
        _drive(60, running=1.0), CRUISE, CellState(), RATES,
        cranked=True, health=1.0,
    )
    worn = run_trip(
        _drive(60, running=1.0), CRUISE, CellState(), RATES,
        cranked=True, health=0.5,
    )
    assert healthy.steps[0].r_int_ohm != worn.steps[0].r_int_ohm
    assert healthy.damage.corrosion_equivalent_h != worn.damage.corrosion_equivalent_h


def test_bay_temperature_carries_forward_across_the_trip_soak_boundary():
    # Seeding the next BayTemperature from the battery's own temperature
    # (which lags the bay by hours) makes the next trip start with a bay
    # spuriously close to the battery instead of where the bay actually was.
    # Stationary/idling so the bay actually runs hot -- see the note in
    # test_the_battery_lags_the_bay_by_hours.
    state = CellState(temp_c=30.0)
    trip = run_trip(_drive(600, speed=0.0, coolant=101.0), CRUISE, state, RATES)
    assert trip.state.bay_temp_c is not None
    # The bay ran hot while driving; the battery barely moved off 30 C.
    assert trip.state.bay_temp_c > trip.state.temp_c + 10.0


def test_a_soak_shows_the_heat_soak_transient():
    # The default dt_s must stay well below BAY_TAU_S (60 s): at dt_s ==
    # BAY_TAU_S the lag clamps to alpha = 1.0 and the bay snaps straight to
    # target, so the multi-minute post-shutdown climb never shows up.
    driving_bay_c = bay_target_c(
        coolant_c=101.0, oil_c=101.0 + CRUISING_GAP_C,
        engine_running=1.0, ambient_c=25.0,
    )
    state = CellState(temp_c=90.0, bay_temp_c=driving_bay_c)
    result = run_soak(1800, CRUISE, state, RATES, initial_coolant_c=101.0)
    trace = [step.t_bay_c for step in result.steps]
    peak = max(trace)
    peak_index = trace.index(peak)
    # It rises for at least a few steps after shutdown...
    assert peak_index >= 3
    assert peak > trace[0]
    # ...and then decays, rather than staying pinned at the peak.
    assert trace[-1] < peak
