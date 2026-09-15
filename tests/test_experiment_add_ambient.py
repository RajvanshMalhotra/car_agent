"""Round 9: add a new climate without disturbing the existing seeded scenarios."""

import random

import pytest

from experiment.add_ambient import extension_params
from experiment.build_dataset import (
    AMBIENTS_C,
    LAYUP_DAYS_RANGE,
    LAYUP_GAP_DAYS_RANGE,
    PARASITIC_A_RANGE,
    SAMPLING_SEED,
    SCENARIOS_PER_AMBIENT,
    TRIP_MINUTES_RANGE,
    draw_scenario_params,
)


def _original_draw(rng):
    """The draw order build_dataset.py used for the existing 80 scenarios."""
    return {
        "trip_minutes": rng.uniform(*TRIP_MINUTES_RANGE),
        "layup_days": rng.uniform(*LAYUP_DAYS_RANGE),
        "layup_gap_days": rng.uniform(*LAYUP_GAP_DAYS_RANGE),
        "parasitic_a": rng.uniform(*PARASITIC_A_RANGE),
    }


def test_shared_draw_matches_the_original_draw_order():
    a, b = random.Random(SAMPLING_SEED), random.Random(SAMPLING_SEED)
    for _ in range(5):
        assert draw_scenario_params(a) == _original_draw(b)


def test_extension_continues_the_seeded_sequence_after_the_existing_scenarios():
    existing = len(AMBIENTS_C) * SCENARIOS_PER_AMBIENT
    reference = random.Random(SAMPLING_SEED)
    for _ in range(existing):
        _original_draw(reference)
    expected = [_original_draw(reference) for _ in range(3)]

    added = extension_params(ambient_c=48.0, count=3)
    assert [p["seed"] for p in added] == [existing, existing + 1, existing + 2]
    assert [p["scenario"] for p in added] == ["amb48_s0080", "amb48_s0081", "amb48_s0082"]
    assert all(p["ambient_c"] == 48.0 for p in added)
    for got, want in zip(added, expected):
        assert {k: got[k] for k in want} == want


def test_extension_rejects_a_climate_the_dataset_already_has():
    with pytest.raises(ValueError):
        extension_params(ambient_c=42.0, count=1)
