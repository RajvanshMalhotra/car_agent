"""The damage rules the planner descends on, checked against the real physics."""

import pytest

torch = pytest.importorskip("torch")

from battery.corrosion import corrosion_rate as reference_rate  # noqa: E402
from cell.aging import FULL_SOC, LOW_SOC, AgingRates, AgingState, Damage, accumulate  # noqa: E402
from experiment.damage import (  # noqa: E402
    PEAK_TEMPERATURE_WEIGHT,
    accumulate_crystal,
    blended_rate,
    corrosion_rate,
    daily_damage,
    health_loss,
    hours_from_actions,
    soft_daily_damage,
)


def test_corrosion_rate_matches_the_project_physics():
    for celsius in (-20.0, 0.0, 25.0, 42.0, 60.0):
        ours = corrosion_rate(torch.tensor([celsius], dtype=torch.float64)).item()
        assert ours == pytest.approx(reference_rate(celsius), rel=1e-9)


def test_corrosion_rate_is_one_at_the_reference_temperature():
    assert corrosion_rate(torch.tensor([25.0], dtype=torch.float64)).item() == pytest.approx(1.0)


def test_blended_rate_sits_between_the_mean_and_peak_rates():
    mean, peak = torch.tensor([30.0]), torch.tensor([50.0])
    blended = blended_rate(mean, peak).item()
    assert corrosion_rate(mean).item() < blended < corrosion_rate(peak).item()
    expected = (1 - PEAK_TEMPERATURE_WEIGHT) * corrosion_rate(mean) + PEAK_TEMPERATURE_WEIGHT * corrosion_rate(peak)
    assert blended == pytest.approx(expected.item())


def test_a_swinging_day_does_more_damage_than_a_flat_one():
    """Heat damage is exponential, so it does not average -- the whole reason for the blend."""
    flat = daily_damage(torch.tensor([30.0]), torch.tensor([1.0]), torch.tensor([24.0]),
                        t_bat_max=torch.tensor([30.0]))
    swinging = daily_damage(torch.tensor([30.0]), torch.tensor([1.0]), torch.tensor([24.0]),
                            t_bat_max=torch.tensor([50.0]))
    assert swinging["corrosion_equivalent_h"] > flat["corrosion_equivalent_h"]


def test_low_soc_hours_only_count_days_below_the_threshold():
    hours = torch.tensor([10.0, 10.0])
    soc = torch.tensor([LOW_SOC - 0.05, LOW_SOC + 0.05])
    assert daily_damage(torch.tensor([25.0, 25.0]), soc, hours)["low_soc_hours"].tolist() == [10.0, 0.0]


def test_full_charge_hours_only_count_days_above_the_threshold():
    hours = torch.tensor([10.0, 10.0])
    soc = torch.tensor([FULL_SOC + 0.01, FULL_SOC - 0.01])
    assert daily_damage(torch.tensor([25.0, 25.0]), soc, hours)["full_charge_hours"].tolist() == [10.0, 0.0]


def test_soft_version_agrees_with_the_hard_one_away_from_the_threshold():
    """Far from a threshold the sigmoid saturates, so soft and hard agree.

    "Far" means several sigmoid widths (SOC_SOFTNESS = 0.02). An earlier
    version of this test used 0.999 as "far above" FULL_SOC = 0.98 -- only one
    width away, where the soft version correctly counts a fraction of the day
    and the hard version counts all of it.
    """
    t, hours = torch.tensor([30.0, 30.0]), torch.tensor([12.0, 12.0])
    soc = torch.tensor([LOW_SOC - 0.20, FULL_SOC + 0.15])
    hard = daily_damage(t, soc, hours, t_bat_max=t)
    soft = soft_daily_damage(t, soc, hours, t_bat_max=t)
    for key in hard:
        torch.testing.assert_close(hard[key], soft[key], atol=1e-3, rtol=1e-3)


def test_soft_version_counts_a_fraction_right_at_the_threshold():
    """Exactly on the threshold the day counts half -- that is what makes it differentiable."""
    t, hours = torch.tensor([30.0]), torch.tensor([12.0])
    soft = soft_daily_damage(t, torch.tensor([LOW_SOC]), hours, t_bat_max=t)
    assert soft["low_soc_hours"].item() == pytest.approx(6.0, rel=1e-6)


def test_gradients_reach_the_dials_through_the_soft_version():
    minutes = torch.tensor([[8.0, 8.0]], requires_grad=True)
    hours = minutes / 60.0 + 8.0
    soc = 0.95 - 0.01 * minutes          # longer trips drain more, as a stand-in
    t_bat = 30.0 + 0.2 * minutes
    d = soft_daily_damage(t_bat, soc, hours, t_bat_max=t_bat + 5.0)
    crystal = accumulate_crystal(torch.zeros(1), d["low_soc_hours"], d["full_charge_hours"])
    health_loss(d["corrosion_equivalent_h"].sum(1), crystal).sum().backward()
    assert torch.isfinite(minutes.grad).all()
    assert minutes.grad.abs().sum() > 0


def test_crystal_growth_matches_cell_aging_step_for_step():
    rates = AgingRates()
    low = torch.tensor([[6.0, 9.0, 3.0]], dtype=torch.float64)
    full = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
    ours = accumulate_crystal(torch.tensor([0.2], dtype=torch.float64), low, full, rates).item()

    state = AgingState(crystal=0.2)
    for day in range(3):
        state = accumulate(state, Damage(
            low_soc_hours=low[0, day].item(), full_charge_hours=full[0, day].item(),
        ), rates)
    assert ours == pytest.approx(state.crystal, rel=1e-9)


def test_crystal_growth_slows_as_the_crystal_grows():
    low = torch.tensor([[50.0]], dtype=torch.float64)
    full = torch.zeros(1, 1, dtype=torch.float64)
    small = accumulate_crystal(torch.tensor([0.1], dtype=torch.float64), low, full)
    large = accumulate_crystal(torch.tensor([0.9], dtype=torch.float64), low, full)
    assert (small - 0.1).item() > (large - 0.9).item()


def test_hours_come_from_the_action_row():
    is_layup = torch.tensor([1.0, 0.0])
    driving_minutes = torch.tensor([0.0, 30.0])
    soak_hours = torch.tensor([24.0, 16.0])
    hours = hours_from_actions(is_layup, driving_minutes, soak_hours)
    assert hours[0].item() == pytest.approx(24.0)          # parked: the whole day
    assert hours[1].item() == pytest.approx(0.5 + 16.0)    # driving: trips plus their soaks


def test_health_loss_grows_with_both_pathways():
    base = health_loss(torch.tensor([1000.0]), torch.tensor([0.1]))
    hotter = health_loss(torch.tensor([2000.0]), torch.tensor([0.1]))
    more_sulfated = health_loss(torch.tensor([1000.0]), torch.tensor([0.5]))
    assert hotter > base and more_sulfated > base
