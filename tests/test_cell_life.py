import dataclasses

import pytest

from cell.aging import AgingRates, Damage
from cell.life import TripSchedule, UnfittedScale, ensemble, project

RATES = AgingRates()
SCHEDULE = TripSchedule()


def _daily(corrosion_h, low_soc_h=0.0):
    """A day's damage that does not depend on health."""
    def day_damage(health):
        return dataclasses.replace(
            Damage.zero(),
            duration_s=86400.0,
            corrosion_equivalent_h=corrosion_h,
            low_soc_hours=low_soc_h,
            full_charge_hours=max(0.0, 20.0 - low_soc_h),
        )
    return day_damage


#: A day of ordinary use in Arrhenius-weighted hours; see cell/aging.py.
ORDINARY_DAY_H = 78.7


def test_ordinary_use_reaches_end_of_life_inside_the_literature_band():
    # SLI batteries last 3-5 years. A model saying 12 is broken.
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    assert estimate.eol_days is not None
    assert 3 * 365 <= estimate.eol_days <= 5 * 365


def test_a_hotter_bay_shortens_life():
    cool = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    hot = project(_daily(ORDINARY_DAY_H * 3.09), RATES, SCHEDULE)
    assert hot.eol_days < cool.eol_days


def test_short_trips_shorten_life_through_sulfation():
    healthy = project(_daily(ORDINARY_DAY_H, low_soc_h=0.0), RATES, SCHEDULE)
    sulfating = project(_daily(ORDINARY_DAY_H, low_soc_h=18.0), RATES, SCHEDULE)
    assert sulfating.eol_days < healthy.eol_days


def test_a_battery_that_survives_the_horizon_reports_no_end_of_life():
    estimate = project(_daily(0.001), RATES, dataclasses.replace(
        SCHEDULE, horizon_days=365.0))
    assert estimate.eol_days is None


def test_the_health_curve_starts_at_one_and_falls():
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    days, healths = zip(*estimate.soh_curve)
    assert healths[0] == pytest.approx(1.0, abs=0.01)
    assert healths[-1] < healths[0]
    assert list(days) == sorted(days)


def test_an_unfitted_estimate_refuses_to_give_a_headline_figure():
    # No battery in this project has reached end of life, so there is nothing
    # to fit the absolute scale against.
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    assert estimate.scale_unfitted is True
    with pytest.raises(UnfittedScale, match="fitted"):
        estimate.headline_days()


def test_a_fitted_estimate_gives_one():
    fitted = dataclasses.replace(RATES, fitted=True)
    estimate = project(_daily(ORDINARY_DAY_H), fitted, SCHEDULE)
    assert estimate.scale_unfitted is False
    assert estimate.headline_days() > 0.0


def test_ratios_are_allowed_even_when_the_scale_is_unfitted():
    # The unfitted constant cancels, which is why the relative claims carry
    # much less validation burden than the absolute one.
    cool = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    hot = project(_daily(ORDINARY_DAY_H * 3.09), RATES, SCHEDULE)
    assert cool.ratio_to(hot) > 2.5


def test_the_ensemble_brackets_the_point_estimate():
    point = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    spread = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=7)
    low, high = spread.interval_days
    assert low < point.eol_days < high


def test_the_ensemble_interval_is_parameter_uncertainty_not_a_world_model():
    spread = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=7)
    assert "parameter" in spread.uncertainty_source
    assert "latent" not in spread.uncertainty_source


def test_the_ensemble_is_reproducible_from_its_seed():
    a = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=3)
    b = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=3)
    assert a.interval_days == b.interval_days


def test_damage_is_recomputed_when_health_has_drifted():
    calls = []

    def day_damage(health):
        calls.append(health)
        return dataclasses.replace(
            Damage.zero(), duration_s=86400.0, corrosion_equivalent_h=78.7
        )

    project(day_damage, RATES, SCHEDULE, resim_threshold=0.02)
    # Resistance rises as health falls, which changes current and temperature,
    # so the per-trip damage cannot be computed once and reused forever.
    assert len(calls) > 1
    assert calls == sorted(calls, reverse=True)
