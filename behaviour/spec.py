"""Driving-behaviour parameters, bounds-checked on construction.

Validation lives here rather than in the LLM prompt: an LLM asked for
"aggressive" will happily emit sustained 0.9g braking. Physics guardrails begin
at behaviour generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar


class SpecValidationError(ValueError):
    """A BehaviourSpec field fell outside its physically plausible range."""


# field -> (minimum, maximum), inclusive. Sized for a conventional ICE
# passenger car on dry tarmac.
BOUNDS: dict[str, tuple[float, float]] = {
    # --- driving style: consumed by the controller ---
    "target_speed_factor": (0.5, 1.3),  # fraction of the posted limit
    "accel_limit_mps2": (0.5, 4.0),  # ~0.4g, generous for a road car
    "decel_limit_mps2": (1.0, 8.0),  # ~0.8g, tyre-limited on dry tarmac
    "jerk_limit_mps3": (0.5, 15.0),
    "following_distance_s": (0.5, 4.0),
    "corner_speed_factor": (0.4, 1.2),
    "reaction_lag_s": (0.0, 2.0),
    "erraticness": (0.0, 1.0),
    # --- trip and thermal: the SLI aging drivers ---
    "trip_duration_s": (60.0, 7200.0),
    "idle_fraction": (0.0, 1.0),
    "hvac_setting": (0.0, 1.0),
    "ambient_temp_c": (-30.0, 55.0),
}


@dataclass(frozen=True)
class BehaviourSpec:
    """One named driving behaviour, expressed as controller parameters."""

    name: str
    target_speed_factor: float
    accel_limit_mps2: float
    decel_limit_mps2: float
    jerk_limit_mps3: float
    following_distance_s: float
    corner_speed_factor: float
    reaction_lag_s: float
    erraticness: float
    trip_duration_s: float
    idle_fraction: float
    hvac_setting: float
    ambient_temp_c: float
    cold_start: bool
    start_stop_enabled: bool

    #: excluded from spec_hash -- renaming a behaviour must not invalidate a
    #: cached run keyed on (spec_hash, scenario, seed).
    UNHASHED_FIELDS: ClassVar[frozenset[str]] = frozenset({"name"})

    def __post_init__(self) -> None:
        violations = []
        if not self.name:
            violations.append("name: must be a non-empty string")
        for field_name, (low, high) in BOUNDS.items():
            value = getattr(self, field_name)
            if not low <= value <= high:
                violations.append(f"{field_name}: {value} outside [{low}, {high}]")
        if violations:
            raise SpecValidationError("; ".join(violations))

    @property
    def spec_hash(self) -> str:
        payload = {
            key: value
            for key, value in sorted(self.to_dict().items())
            if key not in self.UNHASHED_FIELDS
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BehaviourSpec":
        known = {f.name for f in fields(cls)}
        unexpected = set(data) - known
        if unexpected:
            raise SpecValidationError(f"unknown fields: {sorted(unexpected)}")
        missing = known - set(data)
        if missing:
            raise SpecValidationError(f"missing fields: {sorted(missing)}")
        return cls(**data)
