"""Write-ahead log (JSONL decision log) and PendingStore.

Complies with contract §3.6, §4.4, FR-20..FR-24, ADR-0011, ADR-0016.
"""

import json
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from azuma_router.quota import QuotaLedger

from azuma_router.config import Config
from azuma_router.errors import JournalWriteError, LedgerEventNotFoundError
from azuma_router.types import Pending


class DecisionLog:
    """Append-only JSONL log writer with max-byte rotation owned by Router (ADR-0011 / MF-2)."""

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
        except JournalWriteError:
            raise
        except Exception as err:
            raise JournalWriteError(f"WAL append failed: {err}") from err

    def needs_rotation(self) -> bool:
        """True once the active segment exceeds max_bytes; the file is always open, so it exists."""
        self.file.flush()
        return self.path.stat().st_size > self.max_bytes

    def rotate(self, carry_forward: Sequence[dict[str, Any]] = ()) -> None:
        """Moves the active segment to decisions.1.jsonl and re-journals live selects (FR-29)."""
        try:
            self._rotate_files(carry_forward)
        except JournalWriteError:
            raise
        except Exception as err:
            raise JournalWriteError(f"WAL rotation failed: {err}") from err

    def _rotate_files(self, carry_forward: Sequence[dict[str, Any]]) -> None:
        self.file.flush()
        self.file.close()
        rotated_path = self.path.with_name(f"{self.path.stem}.1{self.path.suffix}")
        self.path.replace(rotated_path)
        self.file = self.path.open("a", encoding="utf-8")
        for entry in carry_forward:
            self.append(entry)

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

    def values(self) -> list[Pending]:
        return list(self._pending.values())

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


@dataclass
class _ReplayState:
    """Mutable accumulator threaded through the per-phase replay handlers."""

    snapshot_state_version: int
    ledger: "QuotaLedger | None"
    pending_ttl_s: float
    pending_store: PendingStore = field(default_factory=PendingStore)
    applied_reports: list[tuple[int, np.ndarray, int, float]] = field(default_factory=list)
    applied_ids: set[str] = field(default_factory=set)
    select_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)


def _read_journal(fpath: Path) -> Iterator[dict[str, Any]]:
    """Yields parsed lines; a corrupt final line is a crash artefact and is discarded (FR-24)."""
    with fpath.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            yield json.loads(line_str)
        except json.JSONDecodeError as err:
            if idx == len(lines) - 1:
                return
            raise ValueError(f"WAL corrupt at line {idx + 1} in {fpath}: {err}") from err


def _replay_select(entry: dict[str, Any], rs: _ReplayState) -> None:
    """Every retained select without a later report/expire is a live pending, whatever its
    state_version: selects never advance it (FR-23). Ledger replay is idempotent by event id."""
    d_id = entry["decision_id"]
    if "ledger_event_id" not in entry:
        raise ValueError(f"Missing ledger_event_id in select line for decision {d_id}")
    eid = int(entry["ledger_event_id"])
    rs.select_by_id[d_id] = entry
    if rs.ledger is not None and not rs.ledger.has_event(eid):
        rs.ledger.record(eid, entry["arm_id"], float(entry["est_units"]), float(entry["ts"]))
    rs.pending_store.put(
        Pending(
            decision_id=d_id,
            arm_index=entry["arm_index"],
            x=entry["x"],
            propensity=entry["propensity"],
            ts_select=entry["ts"],
            est_units=entry["est_units"],
            ledger_event_id=eid,
            state_version=entry["state_version"],
            select_entry=entry,
        )
    )


def _replay_report(entry: dict[str, Any], rs: _ReplayState) -> None:
    """Closes the pending; re-applies ledger correction and update only past the snapshot (D15)."""
    d_id = entry["decision_id"]
    select_entry = rs.select_by_id.get(d_id)
    if select_entry is None:
        raise ValueError(f"WAL report for decision {d_id} has no retained select line")
    rs.pending_store.pop(d_id)
    rs.pending_store.mark_applied(d_id)
    rs.applied_ids.add(d_id)
    state_version = int(entry["state_version"])
    if state_version <= rs.snapshot_state_version:
        return
    if rs.ledger is not None:
        try:
            rs.ledger.correct(
                int(select_entry["ledger_event_id"]),
                select_entry["arm_id"],
                float(entry["actual_units"]),
            )
        except LedgerEventNotFoundError as err:
            raise ValueError(f"WAL report for decision {d_id}: {err}") from err
    x_arr = np.array(select_entry["x"], dtype=np.float64)
    rs.applied_reports.append(
        (state_version, x_arr, int(select_entry["arm_index"]), float(entry["reward"]))
    )


def _replay_expire(entry: dict[str, Any], rs: _ReplayState) -> None:
    d_id = entry["decision_id"]
    rs.pending_store.pop(d_id)
    rs.pending_store.mark_expired(d_id, float(entry["ts_select"]) + rs.pending_ttl_s)


_PHASE_HANDLERS = {
    "select": _replay_select,
    "report": _replay_report,
    "expire": _replay_expire,
}


def replay_log(
    active_path: Path,
    rotated_path: Path | None,
    snapshot_state_version: int,
    config: Config,
    ledger: "QuotaLedger | None" = None,
) -> tuple[PendingStore, list[tuple[int, np.ndarray, int, float]], set[str]]:
    """Replays the rotated then the active journal to rebuild PendingStore, the ledger events
    not yet in the snapshot, and the updates to re-apply (FR-23)."""
    rs = _ReplayState(snapshot_state_version, ledger, config.bandit.pending_ttl_s)
    retained = [p for p in (rotated_path, active_path) if p is not None and p.exists()]
    for fpath in retained:
        for entry in _read_journal(fpath):
            phase = str(entry.get("phase"))
            handler = _PHASE_HANDLERS.get(phase)
            if handler is None:
                raise ValueError(f"WAL line with unknown phase {phase!r} in {fpath}")
            handler(entry, rs)
    return rs.pending_store, rs.applied_reports, rs.applied_ids
