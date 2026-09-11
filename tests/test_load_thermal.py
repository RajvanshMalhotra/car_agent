import pytest

from load.thermal import READS, BayTemperature, bay_target_c


def _sample(coolant=100.0, speed=0.0, load=0.3, running=1.0):
    return {"coolant_c": coolant, "speed_mps": speed,
            "engine_load": load, "engine_running": running}


def test_the_bay_sits_between_ambient_and_coolant():
    target = bay_target_c(coolant_c=100.0, speed_mps=20.0, engine_load=0.3,
                          engine_running=1.0, ambient_c=25.0)
    assert 25.0 < target < 100.0


def test_airflow_cools_the_bay():
    stopped = bay_target_c(100.0, 0.0, 0.3, 1.0, 25.0)
    moving = bay_target_c(100.0, 25.0, 0.3, 1.0, 25.0)
    assert moving < stopped


def test_load_heats_the_bay():
    light = bay_target_c(100.0, 20.0, 0.0, 1.0, 25.0)
    heavy = bay_target_c(100.0, 20.0, 1.0, 1.0, 25.0)
    assert heavy > light


def test_the_bay_never_reads_below_ambient():
    target = bay_target_c(coolant_c=10.0, speed_mps=30.0, engine_load=0.0,
                          engine_running=1.0, ambient_c=25.0)
    assert target >= 25.0


def test_shutting_a_hot_engine_off_makes_the_bay_hotter_not_cooler():
    # The heat soak. Airflow stops, the block is still at 130 C, and the bay
    # climbs. It is the hottest the battery ever gets, and the old model threw
    # it away by setting coupling to zero at shutdown.
    bay = BayTemperature(ambient_c=25.0)
    for _ in range(600):
        bay.step(_sample(coolant=130.0, speed=20.0, load=0.4, running=1.0), dt_s=1.0)
    driving = bay.temperature_c

    bay.step(_sample(coolant=130.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    for _ in range(120):
        bay.step(_sample(coolant=130.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    soaking = bay.temperature_c

    assert soaking > driving + 10.0


def test_the_bay_follows_the_coolant_down_once_the_engine_is_cold():
    bay = BayTemperature(ambient_c=25.0, initial_c=95.0)
    for _ in range(600):
        bay.step(_sample(coolant=25.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    assert bay.temperature_c == pytest.approx(25.0, abs=1.0)


def test_the_bay_lags_rather_than_jumping():
    bay = BayTemperature(ambient_c=25.0, initial_c=25.0)
    bay.step(_sample(coolant=130.0, speed=0.0, load=1.0, running=1.0), dt_s=1.0)
    # One second into a 60 s time constant moves it a little, not all the way.
    assert 25.0 < bay.temperature_c < 35.0


def test_a_zero_or_negative_step_is_refused():
    bay = BayTemperature(ambient_c=25.0)
    with pytest.raises(ValueError, match="dt_s"):
        bay.step(_sample(), dt_s=0.0)


def test_reads_declares_every_column_it_touches():
    assert set(READS) == {"coolant_c", "speed_mps", "engine_load", "engine_running"}
