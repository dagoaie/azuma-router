"""Write-ahead log (JSONL decision log) and PendingStore.

Complies with contract §3.6, §4.4, FR-20..FR-24, ADR-0011, ADR-0016.
"""

import json
import math
import os
import stat
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

if TYPE_CHECKING:
    from azuma_router.quota import QuotaLedger

from azuma_router.config import Config
from azuma_router.domain import Pending
from azuma_router.errors import JournalWriteError, LedgerEventNotFoundError
from azuma_router.featurizer import FEATURIZER_VERSION

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def make_private_dir(directory: Path) -> None:
    """Creates the directory if missing and restricts it to the owner (FR-25, RR7-4)."""
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(PRIVATE_DIR_MODE)


def make_private_file(file_path: Path) -> None:
    """Restricts a file this process just wrote to the owner (FR-25, RR7-4)."""
    file_path.chmod(PRIVATE_FILE_MODE)


def fsync_directory(directory: Path) -> None:
    """Makes directory-entry changes (renames, new files) durable (P3-SF-3)."""
    dir_fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _open_private_wal(path: Path) -> TextIOWrapper:
    """Opens one regular WAL file without following symlinks and enforces mode 0600."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
    fd = os.open(str(path), flags, PRIVATE_FILE_MODE)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"WAL path is not a regular file: {path}")
        os.fchmod(fd, PRIVATE_FILE_MODE)
        return os.fdopen(fd, "a", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


class DecisionLog:
    """Append-only JSONL log writer with max-byte rotation owned by Router (ADR-0011 / MF-2)."""

    def __init__(self, path: str | Path, max_bytes: int = 268435456) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        try:
            if not self.path.parent.exists():
                make_private_dir(self.path.parent)  # fresh state dir must be 0700 (FR-25)
            self.file = _open_private_wal(self.path)
        except OSError as err:
            raise JournalWriteError(f"WAL open failed: {err}") from err

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

    def sync(self) -> None:
        """Makes all appended journal entries durable before snapshot publication."""
        try:
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception as err:
            raise JournalWriteError(f"WAL sync failed: {err}") from err

    def needs_rotation(self) -> bool:
        """True once the active segment exceeds max_bytes; the file is always open, so it exists."""
        try:
            self.file.flush()
            return self.path.stat().st_size > self.max_bytes
        except OSError as err:
            raise JournalWriteError(f"WAL stat failed: {err}") from err

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
        self.file = _open_private_wal(self.path)
        fsync_directory(self.path.parent)  # rename + new active segment durable (P3-SF-3)
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
    snapshot_selection_sequence: int
    ledger: "QuotaLedger | None"
    config: Config
    pending_store: PendingStore = field(default_factory=PendingStore)
    applied_reports: list[tuple[int, np.ndarray, int, float]] = field(default_factory=list)
    applied_ids: set[str] = field(default_factory=set)
    expired_ids: set[str] = field(default_factory=set)
    select_by_id: dict[str, Pending] = field(default_factory=dict)
    select_sources: dict[str, str] = field(default_factory=dict)
    ledger_event_owners: dict[int, str] = field(default_factory=dict)
    rng_key: np.ndarray | None = None
    selection_sequence: int = 0
    current_segment: str = ""


def _reject_json_constant(name: str) -> NoReturn:
    """The writer never emits NaN/Infinity (allow_nan=False); reading one is corruption."""
    raise ValueError(f"non-finite JSON constant {name}")


def _read_journal(fpath: Path) -> Iterator[dict[str, Any]]:
    """Streams parsed lines one at a time, so memory is bounded by the longest line, not the file.

    An unterminated final line that does not parse is a crash artefact and is discarded (FR-24):
    only the last line of a file can lack its newline. Any other malformed line, or a line that
    is not a JSON object, is corruption."""
    with fpath.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                entry = json.loads(line_str, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as err:
                if not line.endswith("\n"):
                    return
                raise ValueError(f"WAL corrupt at line {line_number} in {fpath}: {err}") from err
            except (RecursionError, ValueError) as err:
                raise ValueError(f"WAL corrupt at line {line_number} in {fpath}: {err}") from err
            if not isinstance(entry, dict):
                raise ValueError(f"WAL corrupt at line {line_number} in {fpath}: not a JSON object")
            yield entry


def _decision_id(entry: dict[str, Any]) -> str:
    value = entry.get("decision_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"WAL {entry.get('phase')} line without a decision_id")
    return value


def _finite(value: object, key: str, d_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be a finite number")
    try:
        result = float(value)
    except OverflowError as err:
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be a finite number") from err
    if not math.isfinite(result):
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be a finite number")
    return result


def _finite_number(
    entry: dict[str, Any],
    key: str,
    d_id: str,
    *,
    low: float = -math.inf,
    high: float = math.inf,
    open_low: bool = False,
) -> float:
    """A finite JSON number within [low, high] (or (low, high] when open_low)."""
    value = _finite(entry.get(key), key, d_id)
    if value > high or value < low or (open_low and value == low):
        raise ValueError(f"WAL line for decision {d_id}: {key!r}={value} is out of range")
    return value


def _integer(entry: dict[str, Any], key: str, d_id: str, *, minimum: int) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be an integer >= {minimum}")
    return value


def _require_exact_fields(entry: dict[str, Any], phase: str, fields: set[str]) -> None:
    if set(entry) != fields:
        raise ValueError(f"WAL {phase} line has an invalid schema")


def _finite_vector(value: object, key: str, d_id: str, length: int) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must have {length} components")
    for component in value:
        _finite(component, key, d_id)


def _nonnegative_vector(value: object, key: str, d_id: str, length: int) -> None:
    _finite_vector(value, key, d_id, length)
    assert isinstance(value, list)
    if any(component < 0.0 for component in value):
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be non-negative")


def _bool_vector(value: object, key: str, d_id: str, length: int) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must have {length} components")
    if any(not isinstance(component, bool) for component in value):
        raise ValueError(f"WAL line for decision {d_id}: {key!r} must be a list of booleans")


def _validate_utilization(value: object, d_id: str, config: Config) -> None:
    if not isinstance(value, dict) or set(value) != {arm.arm_id for arm in config.arms}:
        raise ValueError(f"WAL select for decision {d_id}: 'utilization' must be an object")
    for arm in config.arms:
        _finite_vector(value[arm.arm_id], "utilization", d_id, len(arm.windows))


def _context_vector(entry: dict[str, Any], d_id: str, dimension: int) -> list[float]:
    raw = entry.get("x")
    if not isinstance(raw, list) or len(raw) != dimension:
        raise ValueError(f"WAL select for decision {d_id}: 'x' must have {dimension} components")
    return [_finite(component, "x", d_id) for component in raw]


def _arm_index(entry: dict[str, Any], d_id: str, config: Config) -> int:
    """Binds arm_index to arm_id: a journal replayed against a different arm layout must not
    credit a reward to the wrong arm."""
    index = _integer(entry, "arm_index", d_id, minimum=0)
    if index >= len(config.arms) or entry.get("arm_id") != config.arms[index].arm_id:
        raise ValueError(f"WAL select for decision {d_id}: arm_index/arm_id do not match config")
    return index


def _decode_select(entry: dict[str, Any], config: Config) -> Pending:
    """Validates every select field the daemon relies on before it can reach the model."""
    _require_exact_fields(
        entry,
        "select",
        {
            "phase",
            "decision_id",
            "ts",
            "state_version",
            "arm_index",
            "arm_id",
            "propensity",
            "alpha",
            "x",
            "policy",
            "score",
            "exploit",
            "bonus",
            "radius",
            "active",
            "shadow_price",
            "utilization",
            "est_units",
            "ledger_event_id",
            "featurizer_version",
            "selection_sequence",
            "rng_key",
        },
    )
    d_id = _decision_id(entry)
    _finite(entry["alpha"], "alpha", d_id)
    for key in ("policy", "score", "exploit", "bonus", "shadow_price"):
        _finite_vector(entry[key], key, d_id, config.bandit.k_max)
    _nonnegative_vector(entry["radius"], "radius", d_id, config.bandit.k_max)
    _bool_vector(entry["active"], "active", d_id, config.bandit.k_max)
    _validate_utilization(entry["utilization"], d_id, config)
    if _integer(entry, "featurizer_version", d_id, minimum=1) != FEATURIZER_VERSION:
        raise ValueError(f"WAL select for decision {d_id}: unsupported featurizer_version")
    return Pending(
        decision_id=d_id,
        arm_index=_arm_index(entry, d_id, config),
        x=_context_vector(entry, d_id, config.bandit.d),
        propensity=_finite_number(entry, "propensity", d_id, low=0.0, high=1.0, open_low=True),
        ts_select=_finite_number(entry, "ts", d_id),
        est_units=_finite_number(entry, "est_units", d_id, low=0.0),
        ledger_event_id=_integer(entry, "ledger_event_id", d_id, minimum=1),
        state_version=_integer(entry, "state_version", d_id, minimum=0),
        selection_sequence=_integer(entry, "selection_sequence", d_id, minimum=1),
        arm_id=entry["arm_id"],
        select_entry=entry,
    )


def _decode_rng_transition(entry: dict[str, Any], d_id: str) -> tuple[int, np.ndarray]:
    raw_sequence = entry.get("selection_sequence")
    raw_rng_key = entry.get("rng_key")
    if (
        isinstance(raw_sequence, bool)
        or not isinstance(raw_sequence, int)
        or raw_sequence <= 0
        or not isinstance(raw_rng_key, list)
        or len(raw_rng_key) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_rng_key)
        or any(not 0 <= value < 2**32 for value in raw_rng_key)
    ):
        raise ValueError(f"Invalid RNG transition in select line for decision {d_id}")
    return raw_sequence, np.asarray(raw_rng_key, dtype=np.uint32)


def _replay_select(entry: dict[str, Any], rs: _ReplayState) -> None:
    """Every retained select without a later report/expire is a live pending, whatever its
    state_version: selects never advance it (FR-23). Ledger replay is idempotent by event id."""
    pending = _decode_select(entry, rs.config)
    d_id = pending.decision_id
    prior_select = rs.select_by_id.get(d_id)
    if prior_select is not None:
        # Rotation deliberately duplicates each live select in the new active segment. Any other
        # duplicate is corruption: it could rebind a report to different learning inputs.
        if (
            rs.current_segment != "active"
            or rs.select_sources[d_id] != "rotated"
            or d_id in rs.applied_ids
            or d_id in rs.expired_ids
            or entry != prior_select.select_entry
        ):
            raise ValueError(f"WAL select for decision {d_id} conflicts with its earlier select")
        rs.select_sources[d_id] = "active"
        return
    selection_sequence, rng_key = _decode_rng_transition(entry, d_id)
    if selection_sequence > rs.snapshot_selection_sequence:
        expected_state_version = rs.snapshot_state_version + len(rs.applied_reports)
        if pending.state_version != expected_state_version:
            msg = (
                f"WAL select for decision {d_id} has state_version {pending.state_version}; "
                f"expected {expected_state_version}"
            )
            raise ValueError(msg)
        expected = rs.selection_sequence + 1
        if selection_sequence != expected:
            msg = (
                f"WAL select for decision {d_id} has selection_sequence {selection_sequence}; "
                f"expected {expected}"
            )
            raise ValueError(msg)
        rs.rng_key = rng_key
        rs.selection_sequence = selection_sequence
    rs.select_by_id[d_id] = pending
    rs.select_sources[d_id] = rs.current_segment
    owner = rs.ledger_event_owners.get(pending.ledger_event_id)
    if owner is not None and owner != d_id:
        raise ValueError(f"WAL select for decision {d_id} reuses ledger_event_id")
    rs.ledger_event_owners[pending.ledger_event_id] = d_id
    if rs.ledger is not None:
        if rs.ledger.has_event(pending.ledger_event_id):
            if selection_sequence > rs.snapshot_selection_sequence:
                raise ValueError(f"WAL select for decision {d_id} reuses snapshot ledger_event_id")
        else:
            rs.ledger.record(
                pending.ledger_event_id, entry["arm_id"], pending.est_units, pending.ts_select
            )
    rs.pending_store.put(pending)


def _replay_report(entry: dict[str, Any], rs: _ReplayState) -> None:
    """Closes the pending; re-applies ledger correction and update only past the snapshot (D15)."""
    _require_exact_fields(
        entry,
        "report",
        {
            "phase",
            "decision_id",
            "ts",
            "state_version",
            "quality",
            "cost_usd",
            "latency_s",
            "actual_input_tokens",
            "actual_output_tokens",
            "reward",
            "actual_units",
        },
    )
    d_id = _decision_id(entry)
    if d_id in rs.applied_ids or d_id in rs.expired_ids:
        raise ValueError(f"WAL report for decision {d_id} follows a terminal phase")
    select = rs.select_by_id.get(d_id)
    if select is None:
        raise ValueError(f"WAL report for decision {d_id} has no retained select line")
    state_version = _integer(entry, "state_version", d_id, minimum=0)
    _finite_number(entry, "ts", d_id)
    _finite_number(entry, "quality", d_id, low=0.0, high=1.0)
    _finite_number(entry, "cost_usd", d_id, low=0.0)
    _finite_number(entry, "latency_s", d_id, low=0.0)
    _integer(entry, "actual_input_tokens", d_id, minimum=0)
    _integer(entry, "actual_output_tokens", d_id, minimum=0)
    reward = _finite_number(entry, "reward", d_id, low=0.0, high=1.0)
    actual_units = _finite_number(entry, "actual_units", d_id, low=0.0)
    if (
        select.selection_sequence > rs.snapshot_selection_sequence
        and state_version <= rs.snapshot_state_version
    ):
        raise ValueError(f"WAL report for decision {d_id} predates its post-snapshot select")
    rs.pending_store.pop(d_id)
    rs.pending_store.mark_applied(d_id)
    rs.applied_ids.add(d_id)
    if state_version <= rs.snapshot_state_version:
        return
    expected = rs.snapshot_state_version + len(rs.applied_reports) + 1
    if state_version != expected:
        msg = (
            f"WAL report for decision {d_id} has state_version {state_version}; "
            f"expected {expected} (updates missing between snapshot and journal)"
        )
        raise ValueError(msg)
    if rs.ledger is not None:
        try:
            rs.ledger.correct(select.ledger_event_id, select.arm_id, actual_units)
        except LedgerEventNotFoundError as err:
            raise ValueError(f"WAL report for decision {d_id}: {err}") from err
    x_arr = np.array(select.x, dtype=np.float64)
    rs.applied_reports.append((state_version, x_arr, select.arm_index, reward))


def _replay_expire(entry: dict[str, Any], rs: _ReplayState) -> None:
    _require_exact_fields(
        entry, "expire", {"phase", "decision_id", "ts", "state_version", "ts_select"}
    )
    d_id = _decision_id(entry)
    if d_id in rs.applied_ids or d_id in rs.expired_ids:
        raise ValueError(f"WAL expire for decision {d_id} follows a terminal phase")
    if d_id not in rs.select_by_id:
        raise ValueError(f"WAL expire for decision {d_id} has no retained select line")
    _finite_number(entry, "ts", d_id)
    _integer(entry, "state_version", d_id, minimum=0)
    ts_select = _finite_number(entry, "ts_select", d_id)
    rs.pending_store.pop(d_id)
    rs.pending_store.mark_expired(d_id, ts_select + rs.config.bandit.pending_ttl_s)
    rs.expired_ids.add(d_id)


_PHASE_HANDLERS = {
    "select": _replay_select,
    "report": _replay_report,
    "expire": _replay_expire,
}


def replay_log(
    active_path: Path,
    rotated_path: Path | None,
    snapshot_state_version: int,
    snapshot_selection_sequence: int,
    config: Config,
    ledger: "QuotaLedger | None" = None,
) -> tuple[
    PendingStore,
    list[tuple[int, np.ndarray, int, float]],
    set[str],
    np.ndarray | None,
    int,
]:
    """Replays the rotated then the active journal to rebuild PendingStore, the ledger events
    not yet in the snapshot, and the updates to re-apply (FR-23)."""
    rs = _ReplayState(
        snapshot_state_version,
        snapshot_selection_sequence,
        ledger,
        config,
        selection_sequence=snapshot_selection_sequence,
    )
    retained = [
        (source, fpath)
        for source, fpath in (("rotated", rotated_path), ("active", active_path))
        if fpath is not None and fpath.exists()
    ]
    for source, fpath in retained:
        rs.current_segment = source
        for entry in _read_journal(fpath):
            phase = entry.get("phase")
            if not isinstance(phase, str):
                raise ValueError(f"WAL line with invalid phase {phase!r} in {fpath}")
            handler = _PHASE_HANDLERS.get(phase)
            if handler is None:
                raise ValueError(f"WAL line with unknown phase {phase!r} in {fpath}")
            try:
                handler(entry, rs)
            except (KeyError, TypeError, ValueError) as err:
                # Every decode/apply failure is `replay_failed` (FR-23), never a traceback.
                raise ValueError(f"WAL corrupt in {fpath}: {err}") from err
    return (
        rs.pending_store,
        rs.applied_reports,
        rs.applied_ids,
        rs.rng_key,
        rs.selection_sequence,
    )
