"""What had to be done to the car during a run, kind by kind.

These were being counted together as "repairs", which is wrong in a way that
matters. Going off the road is not damage: the car that gets put back is the
same car, with the same panels, the same drag and the same engine load. A
*repair* is different -- it resets damage mid-run, so the thermals before and
after are not comparable, and the run summary needs to say so.

Reporting a recovery as a repair overstates how battered the run was, and
damage is the thing that decides whether a run's thermals can be trusted at
all.
"""

from __future__ import annotations

#: What each action is called when there is more than one of it.
PLURALS = {
    "repair": ("repair", "repairs"),
    "recover": ("recovery", "recoveries"),
    "relocate": ("relocation", "relocations"),
}


class Interventions:
    def __init__(self) -> None:
        self.repairs = 0
        self.recoveries = 0
        self.relocations = 0

    def record(self, action: str) -> None:
        if action == "repair":
            self.repairs += 1
        elif action == "recover":
            self.recoveries += 1
        elif action == "relocate":
            self.relocations += 1

    @property
    def total(self) -> int:
        return self.repairs + self.recoveries + self.relocations

    @property
    def thermally_disturbed(self) -> bool:
        """Whether the run's thermal record has a discontinuity in it.

        Only a repair does that. It zeroes damage the car had been carrying,
        so its drag and engine load step at that moment -- and the corrosion
        figure integrates straight over the step without noticing.
        """
        return self.repairs > 0

    def summary(self) -> str:
        """A phrase for the run report, or "" if nothing happened."""
        parts = []
        for action, count in (
            ("repair", self.repairs),
            ("recover", self.recoveries),
            ("relocate", self.relocations),
        ):
            if count:
                one, many = PLURALS[action]
                parts.append(f"{count} {one if count == 1 else many}")
        return ", ".join(parts)
