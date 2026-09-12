"""Finding targets that are closed-form functions of the features beside them.

This module exists because of a specific failure. The real battery dataset's
target satisfies, exactly on every row:

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 (1 - total_ah_used / 16.069411)

The capacity constant is stable to six decimals. A model reported at ~99.7%
accuracy on that target had learned a division, and no amount of extra data
would have changed the number or demonstrated anything. The synthetic dataset
must not repeat it, and asserting that it does not requires a detector that has
been seen to work -- which is why the load-bearing test here feeds it the
identity above and requires it to fire.

Three relationships are checked: exact linear in one feature, exact linear in
two, and the cumulative-sum identity that produced `total_ah_used`. Degenerate
fits are refused rather than reported: a constant column can be fitted to
anything, and calling that a discovery would bury the real findings in noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

#: What counts as exact. These identities hold to floating-point precision, not
#: approximately, so the threshold can be tight.
EXACT_TOLERANCE = 1e-6

#: A fit needs more rows than coefficients, with room to spare.
MIN_ROWS = 8


@dataclass(frozen=True)
class Finding:
    target: str
    features: tuple[str, ...]
    relation: str
    max_residual: float


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def linear_fit(
    rows: Sequence[dict], target: str, features: tuple[str, ...]
) -> tuple[tuple[float, ...], float] | None:
    """Least squares `target = sum(c_i f_i) + c_0`, with its worst residual.

    None when there are too few rows or the normal equations are singular --
    which is what a constant feature produces.
    """
    if len(rows) < MIN_ROWS:
        return None
    design = [[float(row[f]) for f in features] + [1.0] for row in rows]
    targets = [float(row[target]) for row in rows]
    width = len(features) + 1

    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for vector, value in zip(design, targets):
        for i in range(width):
            rhs[i] += vector[i] * value
            for j in range(width):
                normal[i][j] += vector[i] * vector[j]

    coefficients = _solve(normal, rhs)
    if coefficients is None:
        return None

    worst = 0.0
    for vector, value in zip(design, targets):
        predicted = sum(c * v for c, v in zip(coefficients, vector))
        worst = max(worst, abs(predicted - value))
    return tuple(coefficients), worst


def _scale(rows: Sequence[dict], target: str) -> float:
    values = [abs(float(row[target])) for row in rows]
    return max(1.0, max(values)) if values else 1.0


def find_linear(
    rows: Sequence[dict],
    target: str,
    features: tuple[str, ...],
    tol: float = EXACT_TOLERANCE,
) -> list[Finding]:
    """Exact linear relationships in one or two of `features`."""
    findings: list[Finding] = []
    scale = _scale(rows, target)
    candidates = [(f,) for f in features] + list(combinations(features, 2))
    for subset in candidates:
        fit = linear_fit(rows, target, subset)
        if fit is None:
            continue
        coefficients, worst = fit
        if worst / scale > tol:
            continue
        terms = " + ".join(
            f"{c:.10g}*{f}" for c, f in zip(coefficients, subset)
        )
        findings.append(
            Finding(
                target=target,
                features=subset,
                relation=f"{target} = {terms} + {coefficients[-1]:.10g}",
                max_residual=worst,
            )
        )
    return findings


def find_cumulative(
    rows: Sequence[dict],
    target: str,
    feature: str,
    dt_column: str,
    tol: float = EXACT_TOLERANCE,
) -> Finding | None:
    """Is `target` an affine function of `cumsum(feature * dt)`?

    This is the shape that produced `total_ah_used`, and a plain linear fit on
    the instantaneous columns will not see it.
    """
    if len(rows) < MIN_ROWS:
        return None
    running = 0.0
    augmented = []
    for row in rows:
        running += float(row[feature]) * float(row[dt_column])
        augmented.append({"_cumulative": running, target: float(row[target])})

    fit = linear_fit(augmented, target, ("_cumulative",))
    if fit is None:
        return None
    coefficients, worst = fit
    if worst / _scale(rows, target) > tol:
        return None
    return Finding(
        target=target,
        features=(feature, dt_column),
        relation=(
            f"cumsum({feature}*{dt_column}) -> {target} = "
            f"{coefficients[0]:.10g}*cumsum({feature}*{dt_column})"
            f" + {coefficients[1]:.10g}"
        ),
        max_residual=worst,
    )


def report(
    rows: Sequence[dict],
    targets: tuple[str, ...],
    features: tuple[str, ...],
    dt_column: str | None = None,
    tol: float = EXACT_TOLERANCE,
) -> tuple[Finding, ...]:
    """Every analytically recoverable relationship between targets and features."""
    if not rows:
        return ()
    findings: list[Finding] = []
    for target in targets:
        usable = tuple(f for f in features if f != target)
        findings.extend(find_linear(rows, target, usable, tol))
        if dt_column is not None:
            for feature in usable:
                if feature == dt_column:
                    continue
                found = find_cumulative(rows, target, feature, dt_column, tol)
                if found is not None:
                    findings.append(found)
    return tuple(findings)
