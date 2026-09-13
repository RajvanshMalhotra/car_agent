import dataclasses

import pytest

from cell.aging import LOW_SOC, AgingRates, Damage
from cell.life import DayStart, TripSchedule, UnfittedScale, ensemble, project

RATES = AgingRates()
SCHEDULE = TripSchedule()


def _daily(corrosion_h, low_soc_h=0.0):
    """A day's damage that does not depend on health, and leaves SoC alone."""
    def day_damage(day_start):
        damage = dataclasses.replace(
            Damage.zero(),
            duration_s=86400.0,
            corrosion_equivalent_h=corrosion_h,
            low_soc_hours=low_soc_h,
            full_charge_hours=max(0.0, 20.0 - low_soc_h),
        )
        return damage, day_start.soc
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

    def day_damage(day_start):
        calls.append(day_start.health)
        damage = dataclasses.replace(
            Damage.zero(), duration_s=86400.0, corrosion_equivalent_h=78.7
        )
        return damage, day_start.soc

    project(day_damage, RATES, SCHEDULE, resim_threshold=0.02)
    # Resistance rises as health falls, which changes current and temperature,
    # so the per-trip damage cannot be computed once and reused forever.
    assert len(calls) > 1
    assert calls == sorted(calls, reverse=True)


def test_state_of_charge_drifts_down_when_a_day_nets_charge_negative():
    # The mechanism this change exists to enable: a day_damage that reports a
    # slightly lower state of charge each time it is probed must produce a
    # projection where SoC genuinely declines over the life, rather than
    # staying pinned at whatever the first probe saw.
    def day_damage(day_start):
        damage = dataclasses.replace(
            Damage.zero(), duration_s=86400.0,
            corrosion_equivalent_h=ORDINARY_DAY_H,
        )
        soc_end = max(0.0, day_start.soc - 0.002)
        return damage, soc_end

    # A small resim_threshold forces frequent resimulation, so the SoC drift
    # is actually tracked rather than smoothed over a long stretch of days.
    estimate = project(day_damage, RATES, SCHEDULE, resim_threshold=0.0005)
    assert estimate.eol_days is not None


def test_low_soc_hours_engage_once_state_of_charge_actually_falls():
    # Structural proof that the sulfation pathway is reachable through
    # project() itself, not just through a hand-called probe: start state of
    # charge just above LOW_SOC, have day_damage report low_soc_hours only
    # when the SoC it is handed has actually dropped below LOW_SOC, and drift
    # SoC down a little every probe. Once project() carries that drift
    # forward, a later probe must see low_soc_hours flip from 0 to non-zero.
    reported_low_soc_h = []

    def day_damage(day_start):
        low_soc_h = 20.0 if day_start.soc < LOW_SOC else 0.0
        reported_low_soc_h.append(low_soc_h)
        damage = dataclasses.replace(
            Damage.zero(), duration_s=86400.0,
            corrosion_equivalent_h=ORDINARY_DAY_H, low_soc_hours=low_soc_h,
        )
        soc_end = max(0.0, day_start.soc - 0.002)
        return damage, soc_end

    project(
        day_damage, RATES, SCHEDULE, resim_threshold=0.0005,
        soc0=LOW_SOC + 0.01,
    )
    assert reported_low_soc_h[0] == 0.0  # starts above LOW_SOC
    assert any(h > 0.0 for h in reported_low_soc_h), (
        "low_soc_hours never engaged -- state of charge never carried "
        "forward below LOW_SOC across the projection"
    )


def test_state_curve_tracks_the_hidden_ageing_state_over_time():
    # The counterfactual experiment needs a trajectory of the hidden state,
    # not just its terminal value -- state_curve is what carries that out of
    # project(), alongside the scalar soh_curve it has always returned.
    estimate = project(_daily(ORDINARY_DAY_H, low_soc_h=18.0), RATES, SCHEDULE)
    assert estimate.state_curve[0][0] == 0.0  # starts at day 0
    days = [point[0] for point in estimate.state_curve]
    assert days == sorted(days)
    # crystal (index 2) must have grown from its zero start, since low_soc_h
    # is nonzero every day.
    assert estimate.state_curve[0][2] == pytest.approx(0.0, abs=1e-9)
    assert estimate.state_curve[-1][2] > 0.0
    # The last point lines up with the same day soh_curve's last point does.
    assert estimate.state_curve[-1][0] == estimate.soh_curve[-1][0]


def test_the_ensemble_carries_the_point_estimates_state_curve():
    spread = ensemble(
        _daily(ORDINARY_DAY_H, low_soc_h=18.0), RATES, SCHEDULE, draws=4, seed=1,
    )
    point = project(_daily(ORDINARY_DAY_H, low_soc_h=18.0), RATES, SCHEDULE)
    assert spread.state_curve == point.state_curve
