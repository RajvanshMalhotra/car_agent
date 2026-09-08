"""Named spots on the map.

Somewhere to send the car when it keeps crashing where it is. Repeated crashes
in one place mean the AI cannot cope there -- a tight neighbourhood, a junction
it keeps misjudging -- and repairing it on the spot just repeats the crash. A
stretch of highway is a better place to carry on from.

Chosen by driving there once and saving the spot, because only a person looking
at the map knows which road is the easy one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path as FilePath


#: Where the current level might be named in `get_status`. BeamNG's status
#: shape is undocumented and varies by build, so every plausible key is tried
#: rather than guessed at. Not finding one is not an error -- it only means
#: places cannot be checked against the map, which is how they worked before.
LEVEL_KEYS = ("level", "levelName", "mapName", "map", "mission")


def level_from_status(status: object) -> str:
    """The name of the level `get_status` is describing, or "" if it does not
    say. A path like `/levels/west_coast_usa/info.json` reduces to its name."""
    if not isinstance(status, dict):
        return ""
    for key in LEVEL_KEYS:
        raw = status.get(key)
        if isinstance(raw, str) and raw.strip():
            parts = [part for part in raw.replace("\\", "/").split("/") if part]
            if not parts:
                continue
            # ".../levels/<name>/info.json" -> "<name>"
            if len(parts) > 1 and "." in parts[-1]:
                return parts[-2]
            return parts[-1]
    return ""


@dataclass(frozen=True)
class Place:
    name: str
    x_m: float
    y_m: float
    z_m: float
    saved_at: str = ""
    #: Which map it was saved on, or "" for places saved before this was
    #: recorded, or on a build whose status does not name the level.
    level: str = ""

    def belongs_to(self, level: str) -> bool:
        """Whether driving to this place makes sense on `level`.

        Coordinates are meaningless across maps: `drive_to` snaps to the
        nearest navgraph node, so a place from another map is accepted and
        routed to somewhere arbitrary rather than refused.

        An unknown level on either side is not evidence of a mismatch, and
        refusing to drive on a guess would be worse than driving.
        """
        if not self.level or not level:
            return True
        return self.level == level


def _file(directory: FilePath | str) -> FilePath:
    return FilePath(directory) / "places.json"


def _all(directory: FilePath | str) -> dict:
    path = _file(directory)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_place(
    directory: FilePath | str, name: str, x: float, y: float, z: float,
    level: str = "",
) -> Place:
    directory = FilePath(directory)
    directory.mkdir(parents=True, exist_ok=True)
    places = _all(directory)
    places[name] = {
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "level": level,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _file(directory).write_text(json.dumps(places, indent=2, sort_keys=True))
    return load_place(directory, name)


def load_place(directory: FilePath | str, name: str) -> Place | None:
    stored = _all(directory).get(name)
    if stored is None:
        return None
    return Place(name=name, **stored)


def place_names(directory: FilePath | str) -> list[str]:
    return sorted(_all(directory))
