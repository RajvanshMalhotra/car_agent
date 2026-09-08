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


@dataclass(frozen=True)
class Place:
    name: str
    x_m: float
    y_m: float
    z_m: float
    saved_at: str = ""


def _file(directory: FilePath | str) -> FilePath:
    return FilePath(directory) / "places.json"


def _all(directory: FilePath | str) -> dict:
    path = _file(directory)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_place(
    directory: FilePath | str, name: str, x: float, y: float, z: float
) -> Place:
    directory = FilePath(directory)
    directory.mkdir(parents=True, exist_ok=True)
    places = _all(directory)
    places[name] = {
        "x_m": x,
        "y_m": y,
        "z_m": z,
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
