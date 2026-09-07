"""Engine and thermal channels.

These are the channels the battery layer actually needs. Grid corrosion is the
dominant SLI ageing path and it is driven by under-bonnet temperature -- which
neither BeamNG nor OutGauge reports, so it is estimated here from coolant
temperature, airflow and ambient. The same estimator runs on real telemetry.
"""

import pytest

from sim.engine import EngineModel, EngineState

OPERATING_C = 90.0


def warmed_up(ambient_c=20.0, speed_mps=15.0, throttle=0.3):
    engine = EngineModel(ambient_temp_c=ambient_c)
    engine.start()
    for _ in range(60000):  # 20 minutes at 50 Hz
        engine.step(speed_mps=speed_mps, throttle=throttle, dt=0.02)
    return engine


def test_a_stopped_engine_turns_nothing():
    engine = EngineModel(ambient_temp_c=20.0)
    assert engine.state.rpm == 0.0
    assert not engine.state.engine_on


def test_starting_the_engine_records_a_crank_event():
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    assert engine.state.crank_count == 1
    assert engine.state.engine_on


def test_restarting_counts_a_second_crank():
    # Crank count is the sulfation-relevant channel for stop-start and short trips.
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    engine.stop()
    engine.start()
    assert engine.state.crank_count == 2


def test_a_running_stationary_engine_idles():
    engine = EngineModel(ambient_temp_c=20.0, idle_rpm=800.0)
    engine.start()
    engine.step(speed_mps=0.0, throttle=0.0, dt=0.02)
    assert engine.state.rpm == pytest.approx(800.0)


def test_rpm_rises_with_road_speed():
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    engine.step(speed_mps=5.0, throttle=0.3, dt=0.02)
    slow = engine.state.rpm
    engine.step(speed_mps=12.0, throttle=0.3, dt=0.02)
    assert engine.state.rpm > slow


def test_the_gearbox_keeps_rpm_in_a_sane_band_at_road_speeds():
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    for speed in (8.0, 15.0, 22.0, 30.0):
        engine.step(speed_mps=speed, throttle=0.3, dt=0.02)
        assert 800.0 <= engine.state.rpm <= 4000.0, speed


def test_a_cold_start_begins_at_ambient():
    engine = EngineModel(ambient_temp_c=5.0, cold_start=True)
    engine.start()
    assert engine.state.coolant_temp_c == pytest.approx(5.0)


def test_a_warm_start_begins_near_operating_temperature():
    engine = EngineModel(ambient_temp_c=5.0, cold_start=False)
    engine.start()
    assert engine.state.coolant_temp_c > 70.0


def test_coolant_warms_toward_operating_temperature():
    assert warmed_up().state.coolant_temp_c == pytest.approx(OPERATING_C, abs=2.0)


def test_the_thermostat_caps_coolant_temperature():
    assert warmed_up(ambient_c=45.0, throttle=1.0).state.coolant_temp_c <= OPERATING_C + 5.0


def test_a_cold_ambient_slows_warm_up():
    def coolant_after(ambient_c, seconds):
        engine = EngineModel(ambient_temp_c=ambient_c)
        engine.start()
        for _ in range(int(seconds / 0.02)):
            engine.step(speed_mps=10.0, throttle=0.2, dt=0.02)
        return engine.state.coolant_temp_c

    assert coolant_after(-10.0, 120.0) < coolant_after(30.0, 120.0)


def test_a_short_trip_never_reaches_operating_temperature():
    # This is the sulfation pathway: repeated cranking without a full recharge.
    engine = EngineModel(ambient_temp_c=0.0, cold_start=True)
    engine.start()
    for _ in range(int(180 / 0.02)):  # 3 minutes
        engine.step(speed_mps=8.0, throttle=0.2, dt=0.02)
    assert engine.state.coolant_temp_c < OPERATING_C - 10.0


def test_the_engine_bay_is_hotter_than_ambient():
    engine = warmed_up(ambient_c=20.0)
    assert engine.state.underbonnet_temp_c > 20.0


def test_the_engine_bay_is_cooler_than_the_coolant():
    engine = warmed_up(ambient_c=20.0)
    assert engine.state.underbonnet_temp_c < engine.state.coolant_temp_c


def test_idling_runs_a_hotter_bay_than_cruising_at_the_same_coolant_temperature():
    # The whole mechanism: no airflow through the bay means heat accumulates.
    # This is why a courier idling in traffic ages a battery faster.
    def hold(engine, speed_mps, throttle, seconds=300.0):
        for _ in range(int(seconds / 0.02)):
            engine.step(speed_mps=speed_mps, throttle=throttle, dt=0.02)
        return engine.state.underbonnet_temp_c

    engine = warmed_up(ambient_c=35.0)
    cruising = hold(engine, speed_mps=25.0, throttle=0.3)
    idling = hold(engine, speed_mps=0.0, throttle=0.0)
    assert engine.state.coolant_temp_c == pytest.approx(OPERATING_C, abs=2.0)
    assert idling > cruising + 5.0


def test_a_hotter_ambient_gives_a_hotter_bay():
    cool = warmed_up(ambient_c=10.0).state.underbonnet_temp_c
    hot = warmed_up(ambient_c=45.0).state.underbonnet_temp_c
    assert hot > cool


def test_heavy_throttle_heats_the_bay():
    light = warmed_up(throttle=0.1).state.underbonnet_temp_c
    heavy = warmed_up(throttle=0.9).state.underbonnet_temp_c
    assert heavy > light


def test_a_stopped_engine_lets_the_bay_cool_toward_ambient():
    engine = warmed_up(ambient_c=20.0)
    engine.stop()
    for _ in range(int(3600 / 0.02)):  # an hour
        engine.step(speed_mps=0.0, throttle=0.0, dt=0.02)
    assert engine.state.underbonnet_temp_c == pytest.approx(20.0, abs=3.0)


def test_the_state_is_a_snapshot_not_a_live_reference():
    engine = EngineModel(ambient_temp_c=20.0, idle_rpm=800.0)
    engine.start()
    engine.step(speed_mps=0.0, throttle=0.0, dt=0.02)
    before = engine.read_state()
    engine.step(speed_mps=20.0, throttle=1.0, dt=0.02)
    assert isinstance(before, EngineState)
    assert before.rpm == pytest.approx(800.0)
    assert engine.state.rpm > 800.0


def test_a_measured_coolant_temperature_is_used_verbatim():
    # On the real car BeamNG reports coolant temperature. The model must not
    # integrate over a measurement and quietly drift away from it -- only the
    # bay estimate stays modelled.
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    engine.step(speed_mps=10.0, throttle=0.5, dt=0.02, coolant_temp_c=64.0)
    assert engine.state.coolant_temp_c == pytest.approx(64.0)


def test_a_measured_coolant_temperature_still_drives_the_bay_estimate():
    engine = EngineModel(ambient_temp_c=20.0)
    engine.start()
    for _ in range(30000):
        engine.step(speed_mps=0.0, throttle=0.2, dt=0.02, coolant_temp_c=95.0)
    assert engine.state.coolant_temp_c == pytest.approx(95.0)
    assert engine.state.underbonnet_temp_c > 50.0
