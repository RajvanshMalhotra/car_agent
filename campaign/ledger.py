"""What a campaign has already done.

A campaign is hours of real time on a laptop that belongs to someone else. It
will be interrupted -- a game crash, a closed lid, an update -- and a campaign
that restarts from the beginning every time never finishes. So each cell is
written down the moment it completes, and a restart skips what is already
there.

Written on every completion rather than at the end, because the end is exactly
what an interrupted campaign does not reach. The file is plain JSON keyed by
cell so a person can read it, edit it, or delete one line to force a redo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # -- reading ----------------------------------------------------------

    def _all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            # A ledger truncated by a hard power-off is not worth losing a
            # whole campaign over, but silently starting again would repeat
            # every run. Say so loudly instead.
            raise ValueError(
                f"{self.path} is not readable JSON. It was probably truncated "
                f"mid-write. Move it aside to start over, or repair it to keep "
                f"the runs it lists."
            )

    def done_keys(self) -> set[str]:
        """Cells that finished *and* produced data.

        A run that completed with no rows is not done. Counting it would leave
        a hole in the campaign that nothing ever goes back to fill.
        """
        return {
            key for key, entry in self._all().items()
            if entry.get("ok") and entry.get("rows", 0) > 0
        }

    def why_failed(self, key: str) -> str:
        return str(self._all().get(key, {}).get("error", ""))

    def remaining(self, cells: Sequence[Any]) -> list[Any]:
        done = self.done_keys()
        return [cell for cell in cells if cell.key not in done]

    # -- writing ----------------------------------------------------------

    def _write(self, key: str, entry: dict[str, Any]) -> None:
        entries = self._all()
        entries[key] = dict(entry, at=datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2, sort_keys=True))

    def finished(self, key: str, csv: str, rows: int, **extra: Any) -> None:
        self._write(key, {"ok": True, "csv": str(csv), "rows": int(rows), **extra})

    def failed(self, key: str, error: str) -> None:
        self._write(key, {"ok": False, "error": str(error)})
