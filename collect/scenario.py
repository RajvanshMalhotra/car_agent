"""What to drive, and under what conditions.

This replaces `BehaviourSpec`. The LLM behaviour agent is gone: BeamNG's AI
accepts one scalar, `aggression`, so eight generated parameters were collapsed
to one on the way in, and driving diversity now comes from counterfactual
rollouts rather than from the recorded drives.

Two fields are not simulated by the game, and are declared here so the log says
what was assumed. `accessory_load_a` is the draw of lights, blower and ECU,
which BeamNG does not model at all. `ambient_temp_c` reads out of the game but
cannot be set in it, so it is an assumption about the world the drive
represents rather than a property of the drive.

Bounds are checked here rather than trusted from a caller: a plausible-looking
number outside physical range produces a run that looks fine and means nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

#: BeamNG's AI scale. Below ~0.25 the car crawls; above ~1.2 it drives like it
#: is being chased.
AGGRESSION_RANGE = (0.25, 1.2)
#: Lights, blower, ECU, fuel pump. A crank is hundreds of amps and is not this.
ACCESSORY_LOAD_RANGE_A = (0.0, 120.0)
AMBIENT_RANGE_C = (-40.0, 60.0)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    aggression: float = 0.6
    minutes: float = 15.0
    accessory_load_a: float = 35.0
    ambient_temp_c: float = 25.0
    vehicle_config: str = ""
    route: str = "span"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        for field, (low, high) in (
            ("aggression", AGGRESSION_RANGE),
            ("accessory_load_a", ACCESSORY_LOAD_RANGE_A),
            ("ambient_temp_c", AMBIENT_RANGE_C),
        ):
            value = getattr(self, field)
            if not low <= value <= high:
                raise ValueError(f"{field}: {value} outside [{low}, {high}]")
        if self.minutes <= 0:
            raise ValueError(f"minutes: {self.minutes} must be positive")

    @property
    def scenario_hash(self) -> str:
        """Content address. The name is excluded: it labels, it does not define."""
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioSpec":
        return cls(**data)
