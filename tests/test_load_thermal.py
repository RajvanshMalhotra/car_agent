import pytest

from load.thermal import READS, BayTemperature, bay_target_c


def _sample(coolant=100.0, oil=None, running=1.0):
    # A small, realistic cruising gap by default (see the measured table in
    # the module docstring) unless a test asks for something else.
    oil_c = coolant + 0.5 if oil is None else oil
    return {"coolant_c": coolant, "oil_c": oil_c, "engine_running": running}


def test_the_bay_sits_between_ambient_and_coolant():
    target = bay_target_c(coolant_c=100.0, oil_c=100.5, engine_running=1.0,
                          ambient_c=25.0)
    assert 25.0 < target < 100.5


def test_a_stagnant_gap_gives_a_hotter_bay_than_a_ventilated_one():
    # Same coolant temperature; only the measured oil-coolant gap differs.
    ventilated = bay_target_c(coolant_c=90.0, oil_c=90.5, engine_running=1.0,
                              ambient_c=25.0)
    stagnant = bay_target_c(coolant_c=90.0, oil_c=110.0, engine_running=1.0,
                            ambient_c=25.0)
    assert stagnant > ventilated


def test_load_heats_the_bay_via_the_oil_reading():
    # Load is no longer a separate gain term: a harder-working engine shows
    # up as a hotter oil reading (a bigger gap), which is what now carries
    # the load effect.
    light = bay_target_c(coolant_c=90.0, oil_c=90.3, engine_running=1.0,
                         ambient_c=25.0)
    heavy = bay_target_c(coolant_c=90.0, oil_c=105.0, engine_running=1.0,
                         ambient_c=25.0)
    assert heavy > light


def test_the_bay_never_reads_below_ambient():
    target = bay_target_c(coolant_c=10.0, oil_c=10.5, engine_running=1.0,
                          ambient_c=25.0)
    assert target >= 25.0


def test_the_hot_anchor_is_the_hotter_of_the_two_measured_nodes():
    # Engine-off coupling is a fixed constant, independent of the gap, so
    # this isolates anchor selection (max of the two) from the gap-driven
    # coupling used while running.
    oil_hotter = bay_target_c(coolant_c=40.0, oil_c=90.0, engine_running=0.0,
                              ambient_c=20.0)
    coolant_hotter = bay_target_c(coolant_c=90.0, oil_c=40.0, engine_running=0.0,
                                  ambient_c=20.0)
    assert oil_hotter == pytest.approx(coolant_hotter)
    # Anchored to 90 either way, not clipped down to 40.
    assert oil_hotter > 60.0


def test_the_hot_anchor_follows_oil_when_the_engine_is_running_and_oil_leads():
    target = bay_target_c(coolant_c=40.0, oil_c=90.0, engine_running=1.0,
                          ambient_c=20.0)
    # A 50 C gap is deep into the stagnant end of the BOUNDED range, so the
    # coupling sits at (or very near) COUPLING_STAGNANT -- pulled up towards
    # the oil reading relative to the ventilated case, but nowhere near the
    # oil value itself: the ceiling exists precisely so this does not track
    # the hot anchor one-for-one.
    assert 45.0 < target < 60.0


def test_the_bay_never_exceeds_the_hot_anchor():
    # The physical floor under this whole estimator: air surrounded by metal
    # at temperature T cannot exceed T. Checked across engine on/off and
    # across which of coolant/oil leads, including the gap-saturated case
    # that used to let the coupling approach 1.0.
    cases = [
        dict(coolant_c=90.0, oil_c=90.5, engine_running=1.0, ambient_c=25.0),
        dict(coolant_c=90.0, oil_c=212.0, engine_running=1.0, ambient_c=25.0),
        dict(coolant_c=212.0, oil_c=90.0, engine_running=1.0, ambient_c=25.0),
        dict(coolant_c=130.0, oil_c=128.8, engine_running=0.0, ambient_c=25.0),
        dict(coolant_c=130.0, oil_c=138.4, engine_running=0.0, ambient_c=25.0),
    ]
    for case in cases:
        target = bay_target_c(**case)
        assert target <= max(case["coolant_c"], case["oil_c"]) + 1e-9


def test_an_extreme_stagnant_oil_reading_stays_under_the_case_softening_point():
    # This project's own worst measured sample: a stationary high-RPM event
    # drives oil to ~212 C while coolant sits pinned at its 130 C cap, with
    # essentially zero airflow (a large oil-coolant gap). Even there, the bay
    # must stay below the ~130-150 C range where a polypropylene battery case
    # softens -- a bay hot enough to melt the battery it is modelling cannot
    # coexist with a multi-year life prediction.
    target = bay_target_c(coolant_c=130.0, oil_c=212.0, engine_running=1.0,
                          ambient_c=25.0)
    assert target < 130.0


def test_shutting_a_hot_engine_off_makes_the_bay_hotter_not_cooler():
    # The heat soak. Airflow stops, the block is still at 130 C, and the bay
    # climbs. It is the hottest the battery ever gets, and the old model threw
    # it away by setting coupling to zero at shutdown.
    bay = BayTemperature(ambient_c=25.0)
    for _ in range(600):
        bay.step(_sample(coolant=130.0, oil=130.5, running=1.0), dt_s=1.0)
    driving = bay.temperature_c

    # The measured gap goes slightly negative during soak (-1.2 C) -- the
    # heat soak must still rise despite that, not collapse towards ambient.
    bay.step(_sample(coolant=130.0, oil=128.8, running=0.0), dt_s=1.0)
    for _ in range(120):
        bay.step(_sample(coolant=130.0, oil=128.8, running=0.0), dt_s=1.0)
    soaking = bay.temperature_c

    assert soaking > driving + 10.0


def test_the_heat_soak_rises_even_though_the_measured_gap_goes_negative():
    # A direct check of the specific claim: a negative gap must not collapse
    # the engine-off coupling the way it would if the gap drove that regime.
    bay = BayTemperature(ambient_c=25.0, initial_c=60.0)
    trace = []
    for _ in range(180):
        trace.append(bay.step(_sample(coolant=130.0, oil=128.8, running=0.0),
                              dt_s=1.0))
    assert trace[-1] > trace[0]


def test_the_bay_follows_the_coolant_down_once_the_engine_is_cold():
    bay = BayTemperature(ambient_c=25.0, initial_c=95.0)
    for _ in range(600):
        bay.step(_sample(coolant=25.0, oil=23.8, running=0.0), dt_s=1.0)
    assert bay.temperature_c == pytest.approx(25.0, abs=1.0)


def test_the_bay_lags_rather_than_jumping():
    bay = BayTemperature(ambient_c=25.0, initial_c=25.0)
    bay.step(_sample(coolant=130.0, oil=150.0, running=1.0), dt_s=1.0)
    # One second into a 60 s time constant moves it a little, not all the way.
    assert 25.0 < bay.temperature_c < 35.0


def test_a_zero_or_negative_step_is_refused():
    bay = BayTemperature(ambient_c=25.0)
    with pytest.raises(ValueError, match="dt_s"):
        bay.step(_sample(), dt_s=0.0)


def test_reads_declares_every_column_it_touches():
    assert set(READS) == {"coolant_c", "oil_c", "engine_running"}
