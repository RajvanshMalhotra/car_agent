import pytest

from cell.leakage import find_cumulative, find_linear, linear_fit, report

CAPACITY_AH = 16.069411


def _bulb_dataset(n=200):
    """The real dataset's shape: a constant load and a closed-form target.

    Voltage carries a little noise, as the measured channel does. Without it
    voltage is itself an exact function of the target and the detector fires on
    it too -- passing the test for a reason the real data does not share.
    """
    import random
    rng = random.Random(11)
    rows = []
    total = 0.0
    for _ in range(n):
        current, dt_hours = 3.0, 1.0 / 3600.0
        ah = current * dt_hours
        total += ah
        rows.append({
            "current": current,
            "voltage": 12.6 - 0.01 * total + rng.gauss(0.0, 0.002),
            "dt_hours": dt_hours,
            "ah_consumed": ah,
            "total_ah_used": total,
            "True_SoC": 100.0 * (1.0 - total / CAPACITY_AH),
        })
    return rows


def test_it_catches_the_identity_that_caught_us():
    # True_SoC = 100 - (100 / 16.069411) * total_ah_used, exact on every row.
    # This test is the whole point of the module.
    findings = report(
        _bulb_dataset(),
        targets=("True_SoC",),
        features=("current", "voltage", "dt_hours", "total_ah_used"),
    )
    assert any(
        f.target == "True_SoC" and "total_ah_used" in f.features for f in findings
    ), "the detector failed to find the known leak"
    # And it refuses the singular fits: current and dt_hours are constant in
    # this dataset, so nothing may be claimed about them.
    assert all(f.features != ("current",) for f in findings)
    assert all(f.features != ("dt_hours",) for f in findings)


def test_it_catches_the_cumulative_sum_identity():
    finding = find_cumulative(
        _bulb_dataset(), target="total_ah_used", feature="current",
        dt_column="dt_hours", tol=1e-6,
    )
    assert finding is not None
    assert finding.relation.startswith("cumsum")


def test_it_catches_the_product_identity():
    # _bulb_dataset() holds current AND dt_hours both exactly constant, as the
    # real dataset does -- so the product carries zero row-to-row signal and
    # any fit on it is genuinely singular (see the constant-column test
    # below). To observe the product identity at all, one factor must vary;
    # here current does, dt_hours stays the real dataset's constant, and the
    # identity ah_consumed = current * dt_hours still holds exactly.
    import random
    rng = random.Random(7)
    dt_hours = 1.0 / 3600.0
    rows = [{"current": rng.uniform(1.0, 5.0), "dt_hours": dt_hours} for _ in range(50)]
    for row in rows:
        row["ah_consumed"] = row["current"] * row["dt_hours"]
    findings = find_linear(
        rows, target="ah_consumed", features=("current", "dt_hours"), tol=1e-6,
    )
    assert findings


def test_a_genuinely_noisy_relationship_is_not_reported():
    import random
    rng = random.Random(0)
    rows = [
        {"x": rng.uniform(0, 10), "y": rng.uniform(0, 10), "z": rng.uniform(0, 10)}
        for _ in range(200)
    ]
    assert report(rows, targets=("z",), features=("x", "y")) == ()


def test_a_constant_column_does_not_produce_a_spurious_finding():
    # dt_hours is constant in the real dataset. A degenerate fit must be
    # refused, not reported as a discovery.
    rows = [{"k": 1.0, "y": float(i)} for i in range(50)]
    assert linear_fit(rows, target="y", features=("k",)) is None


def test_a_finding_records_how_exact_it_is():
    findings = report(
        _bulb_dataset(), targets=("True_SoC",), features=("total_ah_used",)
    )
    assert findings[0].max_residual < 1e-6


def test_too_few_rows_to_fit_is_not_a_finding():
    rows = [{"x": 1.0, "y": 2.0}]
    assert report(rows, targets=("y",), features=("x",)) == ()


def test_an_empty_dataset_reports_nothing():
    assert report([], targets=("y",), features=("x",)) == ()
