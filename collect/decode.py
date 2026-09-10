"""Turn what the vehicle VM hands back into named samples.

Three shapes arrive on this path and only one of them is the data:

- the "queued" notice, because `run_lua_vehicle` is asynchronous;
- a map of vehicle id to the JSON our Lua built;
- that JSON directly, when the client has already parsed it.

Records travel as positional arrays to keep the payload small -- a named map
per sample at 100 Hz is mostly repeated key strings -- so the schema's order is
load-bearing on both sides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from collect.schema import MEASURED

NAMES = tuple(c.name for c in MEASURED)


class DecodeError(RuntimeError):
    """The payload was not something this can read."""


class _Pending(Exception):
    """The VM has not answered yet. Not a failure."""


@dataclass(frozen=True)
class Drain:
    samples: list[dict] = field(default_factory=list)
    dropped: int = 0
    mass_kg: float = 0.0
    sim_time_s: float = 0.0
    pending: bool = False
    error: str | None = None


def _as_object(payload: object) -> dict:
    if isinstance(payload, str):
        text = payload.strip()
        if "queued" in text:
            raise _Pending()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as error:
            raise DecodeError(f"not JSON: {text[:200]}") from error
    if isinstance(payload, dict):
        # A vehicle-keyed reply: {"73126": "<json>"}. Ours always has 'samples'.
        if "samples" not in payload and "error" not in payload and "ok" not in payload:
            for value in payload.values():
                if isinstance(value, str):
                    return _as_object(value)
            raise DecodeError(f"no payload in {sorted(payload)}")
        return payload
    if payload is None:
        raise _Pending()
    raise DecodeError(f"cannot read a {type(payload).__name__}")


def _record_values(record: object) -> list:
    if isinstance(record, list):
        return record
    if isinstance(record, dict):
        # jsonEncode emits "1".."N" when it will not commit to an array.
        try:
            return [record[str(i + 1)] for i in range(len(record))]
        except KeyError as error:
            raise DecodeError(f"record is not a sequence: {error}") from error
    raise DecodeError(f"record is a {type(record).__name__}")


def decode(payload: object) -> Drain:
    try:
        obj = _as_object(payload)
    except _Pending:
        return Drain(pending=True)

    if obj.get("error") == "not_installed":
        raise DecodeError("the sampler is not installed in this vehicle VM")

    samples = []
    for record in obj.get("samples", []):
        values = _record_values(record)
        if len(values) != len(NAMES):
            raise DecodeError(
                f"record width {len(values)}, schema expects {len(NAMES)}"
            )
        samples.append(dict(zip(NAMES, values)))

    return Drain(
        samples=samples,
        dropped=int(obj.get("dropped") or 0),
        mass_kg=float(obj.get("mass") or 0.0),
        sim_time_s=float(obj.get("t") or 0.0),
        error=obj.get("err"),
    )


def decode_install(payload: object) -> dict:
    """Read the install self-test. Raises if the sampler could not sample."""
    try:
        obj = _as_object(payload)
    except _Pending:
        return {}
    if obj.get("pending"):
        return {}
    if not obj.get("ok", False):
        raise DecodeError(
            f"the sampler could not read the vehicle: {obj.get('error', obj)}"
        )
    return obj
