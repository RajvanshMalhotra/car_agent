"""The behaviour agent: plain English in, a bounds-checked BehaviourSpec out.

The LLM is offline and never in the control loop. It runs once per behaviour,
its answer is validated in code, and the result is cached to disk. Validation
is a gate, not a prompt instruction: an LLM asked for "aggressive" will emit
sustained 0.9g braking however politely the prompt asks it not to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Protocol, Sequence

from behaviour.cache import SpecCache
from behaviour.spec import BOUNDS, BehaviourSpec, SpecValidationError

#: Bump when the prompt changes meaningfully. It is part of the cache key, so a
#: changed prompt regenerates rather than silently reusing older answers.
PROMPT_VERSION = 1

#: Used only when neither the caller nor the client names a model.
FALLBACK_MODEL = "unknown-model"

SYSTEM_PROMPT = """\
You turn a description of a driver into numeric parameters for a vehicle \
controller in a simulator.

The vehicle is a conventional petrol/diesel car with a 12 V lead-acid starter \
battery. The purpose of these runs is to study how driving and trip patterns \
age that battery, so trip length, idle time, ambient temperature and \
electrical accessory use matter as much as how hard the car is driven.

Return one JSON object with exactly these fields and no others. Every numeric \
field must fall inside its stated range -- values outside it are rejected and \
you will be asked again.

{bounds}
  name              a short label for this behaviour
  cold_start        true if the engine starts from cold
  start_stop_enabled  true if the car has stop-start

Notes on the fields:
  target_speed_factor   fraction of the posted speed limit habitually driven
  corner_speed_factor   confidence in bends; 1.0 is an average driver
  reaction_lag_s        delay between seeing the road and acting on it
  erraticness           0 is metronomic, 1 wanders constantly
  idle_fraction         share of the trip spent stationary with the engine on
  hvac_setting          0 is off, 1 is full heating or cooling

Be decisive and take the description seriously: a courier in a hot city really \
does idle a lot with the air conditioning at full, and a nervous learner really \
does crawl. Do not hedge every value toward the middle of its range.\
"""


class BehaviourGenerationError(RuntimeError):
    """The model never produced a spec that passed validation."""


class LLMClient(Protocol):
    """The network seam. Everything else in this package is offline."""

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Return one JSON object matching `schema`."""


def _bounds_block() -> str:
    width = max(len(name) for name in BOUNDS)
    return "\n".join(
        f"  {name:<{width}}  {low} to {high}" for name, (low, high) in BOUNDS.items()
    )


def response_schema() -> dict[str, Any]:
    """JSON schema for the spec, derived from the dataclass and its bounds."""
    properties: dict[str, Any] = {}
    for field in dataclass_fields(BehaviourSpec):
        if field.name in BOUNDS:
            low, high = BOUNDS[field.name]
            properties[field.name] = {
                "type": "number",
                "minimum": low,
                "maximum": high,
            }
        elif field.type in ("bool", bool):
            properties[field.name] = {"type": "boolean"}
        else:
            properties[field.name] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


class BehaviourGenerator:
    def __init__(
        self,
        client: LLMClient,
        cache_dir: Path | str,
        model: str | None = None,
        prompt_version: int = PROMPT_VERSION,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.cache = SpecCache(cache_dir)
        # The model is part of the cache key and the provenance record, so take
        # it from the client that will actually do the work unless pinned.
        self.model = model or getattr(client, "model", FALLBACK_MODEL)
        self.prompt_version = prompt_version
        self.max_attempts = max_attempts
        self.system_prompt = SYSTEM_PROMPT.format(bounds=_bounds_block())

    def cache_key(self, description: str) -> str:
        """Identifies the request, not the answer.

        The model and prompt version are part of it, so changing either
        regenerates instead of silently reusing an answer from a different
        setup.
        """
        payload = json.dumps(
            {
                "description": description,
                "model": self.model,
                "prompt_version": self.prompt_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def generate(self, description: str) -> BehaviourSpec:
        key = self.cache_key(description)
        cached = self.cache.get(key)
        if cached is not None:
            return BehaviourSpec.from_dict(cached)

        spec = self._ask_until_valid(description)
        self.cache.put(
            key,
            spec.to_dict(),
            provenance={
                "description": description,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "spec_hash": spec.spec_hash,
            },
        )
        return spec

    def generate_many(self, descriptions: Sequence[str]) -> list[BehaviourSpec]:
        return [self.generate(description) for description in descriptions]

    def _ask_until_valid(self, description: str) -> BehaviourSpec:
        request = f"Describe this driver as parameters:\n\n{description}"
        schema = response_schema()
        failures: list[str] = []

        for _ in range(self.max_attempts):
            answer = self.client.complete_json(
                system=self.system_prompt, user=request, schema=schema
            )
            try:
                return BehaviourSpec.from_dict(answer)
            except SpecValidationError as error:
                failures.append(str(error))
                request = (
                    f"Describe this driver as parameters:\n\n{description}\n\n"
                    f"Your previous answer was rejected: {error}\n"
                    "Return the whole object again with those fields corrected "
                    "and every other field unchanged."
                )

        raise BehaviourGenerationError(
            f"no valid BehaviourSpec after {self.max_attempts} attempts for "
            f"{description!r}; last rejection: {failures[-1]}"
        )
