"""Which drives a campaign consists of.

The full sweep space -- behaviour x route x ambient x traffic x weather -- is
on the order of a thousand runs at ten real minutes each, which is weeks of
someone else's laptop. So it is sampled rather than gridded: full coverage on
behaviour x ambient, which are the two axes with measured effects, and the rest
held fixed for now.

The order matters as much as the contents, because a campaign is far more
likely to be stopped early than to finish. Ambient moves corrosion about 2.9x
where driving style moves it 1.2-1.5x, so the hot cells carry the most
information and are driven first. A campaign cut short after a third still
answers the main question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Cell:
    """One drive: a behaviour at an ambient temperature."""

    spec_hash: str
    name: str
    ambient_c: float
    repeat: int = 0

    @property
    def key(self) -> str:
        """Stable across restarts, which is what makes the ledger work."""
        base = f"{self.spec_hash}@{self.ambient_c:g}"
        return base if self.repeat == 0 else f"{base}#{self.repeat}"

    @property
    def label(self) -> str:
        return f"{self.name} at {self.ambient_c:.0f} C"


def plan_runs(
    specs: Sequence[Any],
    ambients: Sequence[float],
    repeats: int = 1,
) -> list[Cell]:
    """Every behaviour at every ambient, hottest first."""
    if not specs:
        raise ValueError("a campaign needs at least one behaviour")
    if not ambients:
        raise ValueError("a campaign needs at least one ambient temperature")
    if repeats < 1:
        raise ValueError("a campaign needs at least one repeat of each cell")

    return [
        Cell(spec_hash=spec.spec_hash, name=spec.name,
             ambient_c=float(ambient), repeat=repeat)
        for ambient in sorted(ambients, reverse=True)
        for spec in specs
        for repeat in range(repeats)
    ]
