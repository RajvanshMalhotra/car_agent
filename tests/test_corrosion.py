"""Grid corrosion: the dominant ageing path for an SLI battery in service.

Arrhenius in under-bonnet temperature. The calibration anchor is the standard
rule of thumb -- roughly +10 degC halves service life -- so the model reports
*relative* ageing, in hours-equivalent at a reference temperature. It does not
claim an absolute life in years: nothing here is fitted to a battery that died.
"""

import math

import pytest

from battery.corrosion import (
    ACTIVATION_ENERGY_J_PER_MOL,
    REFERENCE_TEMP_C,
    corrosion_rate,
    equivalent_hours,
)


def test_the_rate_at_the_reference_temperature_is_one():
    assert corrosion_rate(REFERENCE_TEMP_C) == pytest.approx(1.0)


def test_the_rate_doubles_exactly_at_the_calibration_anchor():
    assert corrosion_rate(60.0) / corrosion_rate(50.0) == pytest.approx(2.0, rel=1e-9)


def test_the_doubling_rule_holds_approximately_across_engine_bay_temperatures():
    # Arrhenius cannot double exactly at every temperature -- the ratio falls as
    # temperature rises (2.3x at 20 degC, 1.9x at 65 degC). The "+10 degC halves
    # life" rule is an approximation to this curve, not a law, so it is only
    # honest to hold it loosely and only over the band that actually occurs.
    for base in (35.0, 50.0, 65.0):
        assert 1.8 < corrosion_rate(base + 10.0) / corrosion_rate(base) < 2.2


def test_a_cooler_bay_corrodes_more_slowly():
    assert corrosion_rate(15.0) < corrosion_rate(REFERENCE_TEMP_C)


def test_the_activation_energy_is_physically_plausible_for_grid_corrosion():
    # Literature places lead-acid grid corrosion around 50-70 kJ/mol.
    assert 50_000 < ACTIVATION_ENERGY_J_PER_MOL < 70_000


def test_an_hour_at_the_reference_temperature_is_one_equivalent_hour():
    assert equivalent_hours([REFERENCE_TEMP_C] * 3600, dt_s=1.0) == pytest.approx(1.0)


def test_a_hotter_trip_accumulates_more_damage():
    cool = equivalent_hours([40.0] * 3600, dt_s=1.0)
    hot = equivalent_hours([70.0] * 3600, dt_s=1.0)
    assert hot > cool


def test_thirty_degrees_hotter_is_about_seven_times_the_damage():
    # Not 2^3 = 8: compounding the true Arrhenius curve gives slightly less
    # than three exact doublings.
    ratio = equivalent_hours([80.0] * 100, dt_s=1.0) / equivalent_hours(
        [50.0] * 100, dt_s=1.0
    )
    assert ratio == pytest.approx(7.1, rel=0.05)


def test_an_empty_trace_accumulates_nothing():
    assert equivalent_hours([], dt_s=1.0) == 0.0


def test_damage_scales_with_duration():
    short = equivalent_hours([60.0] * 100, dt_s=1.0)
    long = equivalent_hours([60.0] * 200, dt_s=1.0)
    assert long == pytest.approx(2 * short)


def test_a_brief_spike_matters_less_than_a_long_soak():
    # Corrosion integrates exposure; peak temperature alone is not the story.
    spike = equivalent_hours([50.0] * 3590 + [95.0] * 10, dt_s=1.0)
    soak = equivalent_hours([62.0] * 3600, dt_s=1.0)
    assert soak > spike


def test_the_integral_matches_a_hand_computed_value():
    trace = [30.0, 60.0, 90.0]
    expected = sum(corrosion_rate(t) for t in trace) * 2.0 / 3600.0
    assert equivalent_hours(trace, dt_s=2.0) == pytest.approx(expected)
