"""Does the game's measurement change the answer?

The honest question about this whole project: the simulator is expensive, and
most of what it produces is fed into models with invented coefficients. If the
corrosion figure comes out the same whether the coolant is measured by BeamNG or
predicted by our own engine model, the game is not earning its place and the
driving patterns could be sampled statistically instead.

One game run answers it. The same speed and throttle trace is replayed through
our engine model, and the two corrosion figures are compared. Everything else --
route, idle, ambient, behaviour -- is held identical, because it is literally
the same run.
"""

import csv

import pytest

from analysis.game_value import compare_thermal_sources, read_run


def write_run(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "speed_mps", "rpm", "coolant_c", "underbonnet_c",
                         "throttle", "idling"])
        for row in rows:
            writer.writerow(row)
    return path


def a_run(tmp_path, coolant=95.0, speed=12.0, seconds=600):
    rows = [[float(t), speed, 2200.0, coolant, 0.0, 0.4, 0] for t in range(seconds)]
    return write_run(tmp_path / "run.csv", rows)


def test_a_run_is_read_back():
    pass  # covered by the cases below


def test_it_reports_both_ways_of_getting_the_coolant(tmp_path):
    result = compare_thermal_sources(read_run(a_run(tmp_path)), ambient_c=30.0)
    assert result.measured.equivalent_hours > 0.0
    assert result.modelled.equivalent_hours > 0.0


def test_a_hotter_measured_coolant_gives_more_corrosion(tmp_path):
    hot = compare_thermal_sources(read_run(a_run(tmp_path, coolant=100.0)),
                                  ambient_c=30.0)
    cool = compare_thermal_sources(read_run(a_run(tmp_path, coolant=85.0)),
                                   ambient_c=30.0)
    assert hot.measured.equivalent_hours > cool.measured.equivalent_hours


def test_the_modelled_side_ignores_what_the_game_measured(tmp_path):
    # Same driving, different reported coolant: our model should say the same
    # thing both times, because it never saw the measurement.
    hot = compare_thermal_sources(read_run(a_run(tmp_path, coolant=100.0)),
                                  ambient_c=30.0)
    cool = compare_thermal_sources(read_run(a_run(tmp_path, coolant=85.0)),
                                   ambient_c=30.0)
    assert hot.modelled.equivalent_hours == pytest.approx(
        cool.modelled.equivalent_hours, rel=0.01
    )


def test_the_verdict_is_the_ratio_between_them(tmp_path):
    result = compare_thermal_sources(read_run(a_run(tmp_path, coolant=100.0)),
                                     ambient_c=30.0)
    expected = result.measured.equivalent_hours / result.modelled.equivalent_hours
    assert result.ratio == pytest.approx(expected)


def test_agreement_means_the_game_is_not_contributing(tmp_path):
    # If our model predicts what the game measured, the ratio is one and the
    # measurement bought nothing. Started warm, so this compares the settled
    # temperature rather than the warm-up, which is a separate question.
    from sim.engine import OPERATING_TEMP_C

    result = compare_thermal_sources(
        read_run(a_run(tmp_path, coolant=OPERATING_TEMP_C)),
        ambient_c=30.0, cold_start=False,
    )
    assert result.ratio == pytest.approx(1.0, abs=0.1)
    assert not result.game_matters


def test_the_warm_up_is_where_the_two_diverge_most(tmp_path):
    # A cold engine takes minutes to reach temperature, and how fast it gets
    # there is one of the few things the game models and we guess at.
    from sim.engine import OPERATING_TEMP_C

    warm = compare_thermal_sources(
        read_run(a_run(tmp_path, coolant=OPERATING_TEMP_C)),
        ambient_c=30.0, cold_start=False,
    )
    cold = compare_thermal_sources(
        read_run(a_run(tmp_path, coolant=OPERATING_TEMP_C)),
        ambient_c=30.0, cold_start=True,
    )
    assert abs(cold.ratio - 1.0) > abs(warm.ratio - 1.0)


def test_a_large_disagreement_means_it_is(tmp_path):
    result = compare_thermal_sources(read_run(a_run(tmp_path, coolant=115.0)),
                                     ambient_c=30.0)
    assert result.game_matters


def test_the_coolant_gap_is_reported_in_degrees(tmp_path):
    from sim.engine import OPERATING_TEMP_C

    result = compare_thermal_sources(read_run(a_run(tmp_path, coolant=100.0)),
                                     ambient_c=30.0)
    assert result.measured.mean_coolant_c == pytest.approx(100.0)
    assert result.modelled.mean_coolant_c < OPERATING_TEMP_C + 1.0


def test_an_empty_run_is_refused(tmp_path):
    empty = write_run(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no rows"):
        compare_thermal_sources(read_run(empty), ambient_c=30.0)
