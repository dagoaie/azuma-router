"""Tests for DecisionLog WAL and PendingStore (T16 & T17)."""

import json
from pathlib import Path

import pytest

from azuma_router.errors import JournalWriteError
from azuma_router.types import Pending
from azuma_router.wal import DecisionLog, PendingStore


def test_decision_log_append_flush_ac17(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    dlog = DecisionLog(path=log_path, max_bytes=1000)

    # Spy on flush
    flushed = False
    original_flush = dlog.file.flush

    def spy_flush() -> None:
        nonlocal flushed
        original_flush()
        flushed = True

    dlog.file.flush = spy_flush

    entry = {
        "phase": "select",
        "decision_id": "018f6c2e-0000-7000-8000-000000000001",
        "ts": 1757800000.0,
        "state_version": 0,
        "arm_index": 0,
        "arm_id": "gpt-5-codex@openai-codex",
        "propensity": 0.8123,
        "x": [1.0] * 32,
        "policy": [0.8123, 0.1877],
        "score": [0.71, 0.44],
        "exploit": [0.5, 0.3],
        "bonus": [0.21, 0.14],
        "shadow_price": [0.0, 0.0],
        "utilization": {"gpt-5-codex@openai-codex": [0.0, 0.0]},
        "est_units": 1.0,
        "featurizer_version": 1,
    }

    dlog.append(entry)
    assert flushed is True

    # Read back file content
    with log_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["phase"] == "select"
    assert parsed["decision_id"] == "018f6c2e-0000-7000-8000-000000000001"
    dlog.close()


def test_decision_log_rotation(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    dlog = DecisionLog(path=log_path, max_bytes=50)

    select_entry = {
        "phase": "select",
        "decision_id": "018f6c2e-0000-7000-8000-000000000001",
        "ts": 1757800000.0,
        "state_version": 0,
    }

    dlog.append(select_entry)
    assert dlog.needs_rotation()

    dlog.rotate(carry_forward=[select_entry])

    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists()
    dlog.close()


def test_decision_log_rotate_wraps_io_errors_in_journal_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dlog = DecisionLog(path=tmp_path / "decisions.jsonl", max_bytes=50)
    dlog.append({"phase": "select", "decision_id": "id-1"})

    def failing_replace(self: Path, target: Path) -> Path:
        raise OSError("disk detached")

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(JournalWriteError, match="disk detached"):
        dlog.rotate(carry_forward=[])


def test_decision_log_open_failure_is_a_journal_write_error(tmp_path: Path) -> None:
    """P3-MF-3: an unopenable journal at boot is a typed fatal error, not a raw OSError."""
    (tmp_path / "journal").write_text("a file where the journal directory should be")

    with pytest.raises(JournalWriteError, match="WAL open failed"):
        DecisionLog(tmp_path / "journal" / "decisions.jsonl")


def test_pending_store_sweep() -> None:
    store = PendingStore()
    p1 = Pending(
        decision_id="id-1",
        arm_index=0,
        x=[1.0] * 32,
        propensity=0.8,
        ts_select=100.0,
        est_units=1.0,
        ledger_event_id=1,
        state_version=0,
        selection_sequence=1,
        arm_id="arm-1",
        select_entry={"phase": "select", "decision_id": "id-1"},
    )
    p2 = Pending(
        decision_id="id-2",
        arm_index=1,
        x=[1.0] * 32,
        propensity=0.2,
        ts_select=200.0,
        est_units=1.0,
        ledger_event_id=2,
        state_version=0,
        selection_sequence=2,
        arm_id="arm-2",
        select_entry={"phase": "select", "decision_id": "id-2"},
    )

    store.put(p1)
    store.put(p2)
    assert len(store) == 2

    # Sweep at now = 150.0, ttl = 30.0 -> p1 expired (now - ts = 50 > 30), p2 kept (150 - 200 < 0)
    expired = store.sweep(now=150.0, ttl_s=30.0)
    assert len(expired) == 1
    assert expired[0].decision_id == "id-1"
    assert len(store) == 1
    assert "id-2" in store


def test_expired_ids_eviction_limit_lru() -> None:
    store = PendingStore(max_lru=3)

    store.mark_expired("id-1", 100.0)
    store.mark_expired("id-2", 101.0)
    store.mark_expired("id-3", 102.0)
    store.mark_expired("id-4", 103.0)

    # id-1 (oldest) evicted
    assert not store.is_expired("id-1")
    assert store.is_expired("id-2")
    assert store.is_expired("id-3")
    assert store.is_expired("id-4")


def test_pending_store_keeps_at_most_ten_thousand_expired_ids_by_default() -> None:
    store = PendingStore()
    for n in range(10_001):
        store.mark_expired(f"id-{n}", float(n))

    assert not store.is_expired("id-0")
    assert store.is_expired("id-1")
    assert store.is_expired("id-10000")


def test_rotation_fsyncs_the_journal_directory_after_the_rename(
    tmp_path: Path, fsynced_inodes: list[int]
) -> None:
    """decisions.1.jsonl must survive a power cut right after rotation (P3-SF-3)."""
    dlog = DecisionLog(path=tmp_path / "decisions.jsonl", max_bytes=1)
    dlog.append({"phase": "select", "decision_id": "d1"})

    dlog.rotate(carry_forward=[])

    assert tmp_path.stat().st_ino in fsynced_inodes
    dlog.close()


def test_sync_fsyncs_the_active_journal_before_snapshot_publication(
    tmp_path: Path, fsynced_inodes: list[int]
) -> None:
    dlog = DecisionLog(path=tmp_path / "decisions.jsonl")
    dlog.append({"phase": "select", "decision_id": "d1"})

    dlog.sync()

    assert (tmp_path / "decisions.jsonl").stat().st_ino in fsynced_inodes
    dlog.close()
