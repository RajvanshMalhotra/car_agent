import dataclasses

import pytest

from cell.aging import EOL_SOH, AgingRates, AgingState, Damage, accumulate, soh

RATES = AgingRates()


def _damage(**kwargs):
    return dataclasses.replace(Damage.zero(), **kwargs)


def test_a_new_battery_is_in_perfect_health():
    assert soh(AgingState(), RATES) == pytest.approx(1.0)


def test_corrosion_hours_reduce_health():
    state = accumulate(AgingState(), _damage(corrosion_equivalent_h=5000.0), RATES)
    assert soh(state, RATES) < 1.0


def test_corrosion_damage_is_sublinear_in_time():
    # Schiffer: the corrosion layer grows as roughly t^0.6, so doubling the
    # exposure does less than double the damage.
    once = accumulate(AgingState(), _damage(corrosion_equivalent_h=10000.0), RATES)
    twice = accumulate(AgingState(), _damage(corrosion_equivalent_h=20000.0), RATES)
    loss_once = 1.0 - soh(once, RATES)
    loss_twice = 1.0 - soh(twice, RATES)
    assert loss_twice < 2.0 * loss_once
    assert loss_twice > loss_once


def test_sulfation_growth_slows_as_crystals_get_larger():
    # The rate depends on the current crystal size, which is why this has to be
    # a state and cannot be a running total.
    fresh = accumulate(AgingState(), _damage(low_soc_hours=100.0), RATES)
    sulfated = accumulate(
        AgingState(crystal=0.8), _damage(low_soc_hours=100.0), RATES
    )
    assert (fresh.crystal - 0.0) > (sulfated.crystal - 0.8)


def test_a_full_charge_dissolves_sulfation():
    state = accumulate(
        AgingState(crystal=0.5), _damage(full_charge_hours=200.0), RATES
    )
    assert state.crystal < 0.5


def test_sulfation_stays_inside_its_bounds():
    state = accumulate(AgingState(crystal=0.99), _damage(low_soc_hours=1e6), RATES)
    assert 0.0 <= state.crystal <= 1.0
    state = accumulate(AgingState(crystal=0.01), _damage(full_charge_hours=1e6), RATES)
    assert 0.0 <= state.crystal <= 1.0


def test_vibration_sheds_plate_material():
    state = accumulate(AgingState(), _damage(vibration_dose=1e5), RATES)
    assert state.shedding > 0.0
    assert soh(state, RATES) < 1.0


def test_health_never_goes_negative():
    state = AgingState(corrosion_hours=1e9, crystal=1.0, shedding=1.0)
    assert soh(state, RATES) >= 0.0


def test_damage_adds_componentwise():
    a = _damage(duration_s=10.0, ah_throughput=1.0, low_soc_hours=2.0)
    b = _damage(duration_s=5.0, ah_throughput=0.5, low_soc_hours=1.0)
    total = a + b
    assert total.duration_s == pytest.approx(15.0)
    assert total.ah_throughput == pytest.approx(1.5)
    assert total.low_soc_hours == pytest.approx(3.0)


def test_the_default_rate_constants_are_flagged_unfitted():
    # No battery in this project has reached end of life. Anything built on
    # these constants has to say so.
    assert AgingRates().fitted is False
    assert AgingRates(fitted=True).fitted is True


#: A day of ordinary use, in Arrhenius-weighted hours at 25 C. Derived in the
#: AgingRates docstring: 1.5 h driving with a 60 C bay, 1 h of heat soak near
#: 75 C, and 21.5 h parked at 25 C.
ORDINARY_DAY_H = 78.7


def _years_to_eol(daily_h, rates=RATES, limit_years=40):
    state = AgingState()
    day = 0
    while soh(state, rates) > EOL_SOH and day < limit_years * 365:
        state = accumulate(state, _damage(corrosion_equivalent_h=daily_h), rates)
        day += 1
    return day / 365


def test_the_default_constants_land_inside_the_literature_band():
    # A sanity anchor, not a calibration: SLI batteries last 3-5 years. A model
    # saying 12 is broken.
    years = _years_to_eol(ORDINARY_DAY_H)
    assert 3.0 <= years <= 5.0, f"end of life at {years:.2f} years"


def test_a_hot_climate_is_markedly_worse():
    # 42 C ambient is about 3.1x the weighted exposure of 25 C.
    years = _years_to_eol(ORDINARY_DAY_H * 3.09)
    assert years < _years_to_eol(ORDINARY_DAY_H)
    assert years < 2.0


def test_a_gently_used_garaged_car_is_not_claimed_to_last_forever():
    # The other end of the sanity band. Nothing here may predict 12 years.
    assert _years_to_eol(28.1) < 12.0
