"""Write-ahead log (JSONL decision log) and PendingStore.

Complies with contract §3.6, §4.4, FR-20..FR-24, ADR-0011, ADR-0016.
"""

import json
from collections import OrderedDict
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from azuma_router.quota import QuotaLedger

from azuma_router.config import Config
from azuma_router.errors import JournalWriteError
from azuma_router.types import Pending


class DecisionLog:
    """Append-only JSONL log writer with max-byte rotation (ADR-0011)."""

    def __init__(self, path: str | Path, max_bytes: int = 268435456) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIOWrapper = self.path.open("a", encoding="utf-8")

    def append(self, entry: dict[str, Any]) -> None:
        """Appends a single JSONL line and flushes immediately (FR-20)."""
        try:
            line = json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n"
            self.file.write(line)
            self.file.flush()

            phase = entry.get("phase")
            if phase in ("report", "expire"):
                self._maybe_rotate()
        except JournalWriteError:
            raise
        except Exception as err:
            raise JournalWriteError(f"WAL append failed: {err}") from err

    def _maybe_rotate(self) -> None:
        """Rotates file to decisions.1.jsonl if size exceeds max_bytes."""
        if not self.path.exists():
            return
        if self.path.stat().st_size > self.max_bytes:
            self.file.flush()
            self.file.close()

            # Rotate path: decisions.jsonl -> decisions.1.jsonl
            rotated_path = self.path.with_name(f"{self.path.stem}.1{self.path.suffix}")
            if rotated_path.exists():
                rotated_path.unlink()
            self.path.replace(rotated_path)

            self.file = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()


class PendingStore:
    """In-memory store of pending decisions awaiting report (ADR-0016)."""

    def __init__(self, max_lru: int = 10000) -> None:
        self._pending: OrderedDict[str, Pending] = OrderedDict()
        self._applied_ids: OrderedDict[str, None] = OrderedDict()
        self._expired_ids: OrderedDict[str, float] = OrderedDict()
        self._max_lru = max_lru

    def put(self, pending: Pending) -> None:
        self._pending[pending.decision_id] = pending

    def get(self, decision_id: str) -> Pending | None:
        return self._pending.get(decision_id)

    def pop(self, decision_id: str) -> Pending | None:
        return self._pending.pop(decision_id, None)

    def is_applied(self, decision_id: str) -> bool:
        return decision_id in self._applied_ids

    def mark_applied(self, decision_id: str) -> None:
        self._applied_ids[decision_id] = None
        if len(self._applied_ids) > self._max_lru:
            self._applied_ids.popitem(last=False)

    def is_expired(self, decision_id: str) -> bool:
        return decision_id in self._expired_ids

    def get_expired_at(self, decision_id: str) -> float | None:
        return self._expired_ids.get(decision_id)

    def mark_expired(self, decision_id: str, expired_at: float) -> None:
        self._expired_ids[decision_id] = expired_at
        if len(self._expired_ids) > self._max_lru:
            self._expired_ids.popitem(last=False)

    def sweep(self, now: float, ttl_s: float) -> list[Pending]:
        """Sweeps expired pending decisions in insertion order until first non-expired (FR-22)."""
        expired: list[Pending] = []
        to_delete: list[str] = []

        for d_id, p in self._pending.items():
            if now - p.ts_select > ttl_s:
                expired.append(p)
                to_delete.append(d_id)
                self.mark_expired(d_id, p.ts_select + ttl_s)
            else:
                break

        for d_id in to_delete:
            del self._pending[d_id]

        return expired

    def __len__(self) -> int:
        return len(self._pending)

    def __contains__(self, decision_id: str) -> bool:
        return decision_id in self._pending


def replay_log(
    active_path: Path,
    rotated_path: Path | None,
    snapshot_state_version: int,
    config: Config,
    ledger: "QuotaLedger | None" = None,
) -> tuple[PendingStore, list[tuple[int, np.ndarray, int, float]], set[str]]:
    """Replays JSONL WAL files in order to restore PendingStore and unapplied updates (FR-23)."""
    pending_store = PendingStore()
    applied_reports: list[tuple[int, np.ndarray, int, float]] = []
    applied_ids: set[str] = set()
    select_by_id: dict[str, dict[str, Any]] = {}
    pending_event_ids: dict[str, int] = {}

    files_to_read: list[Path] = []
    if rotated_path and rotated_path.exists():
        files_to_read.append(rotated_path)
    if active_path.exists():
        files_to_read.append(active_path)

    for fpath in files_to_read:
        with fpath.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                entry = json.loads(line_str)
            except Exception as err:
                is_last_line = idx == len(lines) - 1
                if is_last_line:
                    # Corrupt tail at EOF -> discard and stop reading this file
                    break
                msg = f"WAL corrupt at line {idx + 1} in {fpath}: {err}"
                raise ValueError(msg) from err

            phase = entry.get("phase")
            d_id = entry.get("decision_id", "")
            s_ver = entry.get("state_version", 0)

            if phase == "select":
                select_by_id[d_id] = entry
                if s_ver >= snapshot_state_version:
                    eid = 0
                    if ledger is not None:
                        eid = ledger.record(
                            entry["arm_id"],
                            float(entry.get("est_units", 0.0)),
                            float(entry["ts"]),
                        )
                    pending_event_ids[d_id] = eid
                    pending = Pending(
                        decision_id=d_id,
                        arm_index=entry["arm_index"],
                        x=entry["x"],
                        propensity=entry["propensity"],
                        ts_select=entry["ts"],
                        est_units=entry.get("est_units", 0.0),
                        ledger_event_id=eid,
                        state_version=s_ver,
                    )
                    pending_store.put(pending)

            elif phase == "report":
                pending_store.pop(d_id)
                applied_ids.add(d_id)
                pending_store.mark_applied(d_id)
                eid = pending_event_ids.pop(d_id, 0)
                if ledger is not None and eid > 0:
                    arm_id = select_by_id[d_id]["arm_id"] if d_id in select_by_id else ""
                    actual_u = float(entry.get("actual_units", 0.0))
                    if arm_id:
                        ledger.correct(eid, arm_id, actual_u)
                if s_ver > snapshot_state_version:
                    if d_id in select_by_id:
                        s_entry = select_by_id[d_id]
                        x_arr = np.array(s_entry["x"], dtype=np.float64)
                        arm_idx = s_entry["arm_index"]
                        reward = float(entry["reward"])
                        applied_reports.append((s_ver, x_arr, arm_idx, reward))

            elif phase == "expire":
                pending_store.pop(d_id)
                pending_event_ids.pop(d_id, None)
                ts_select = float(entry.get("ts_select", entry.get("ts", 0.0)))
                pending_store.mark_expired(d_id, ts_select + config.bandit.pending_ttl_s)

    return pending_store, applied_reports, applied_ids
