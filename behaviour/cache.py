"""On-disk cache of generated behaviour specs.

Generation is offline and happens once per behaviour. Caching is what makes a
campaign reproducible from (spec_hash, scenario, seed) and what keeps an LLM
call off the critical path of a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SpecCache:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())["spec"]
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"corrupt cache entry at {path}: {error}") from error

    def put(
        self, key: str, spec: dict[str, Any], provenance: dict[str, Any]
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "spec": spec,
            "provenance": {
                **provenance,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        self._path(key).write_text(json.dumps(record, indent=2, sort_keys=True))
