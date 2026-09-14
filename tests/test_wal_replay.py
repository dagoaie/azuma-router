"""Tests for WAL replay (T18 & AC-20)."""

from pathlib import Path

import pytest

from azuma_router.config import load_config
from azuma_router.wal import replay_log


def test_replay_pending_only_ac20() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/pending_only.jsonl")

    pending_store, applied_reports, _applied_ids = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        config=cfg,
    )

    assert len(pending_store) == 1
    assert "018f6c2e-0000-7000-8000-000000000001" in pending_store
    assert len(applied_reports) == 0


def test_replay_reported_ac20() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/reported.jsonl")

    pending_store, applied_reports, applied_ids = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        config=cfg,
    )

    assert len(pending_store) == 0
    assert len(applied_reports) == 1
    assert applied_reports[0][0] == 1  # state_version 1
    assert "018f6c2e-0000-7000-8000-000000000001" in applied_ids


def test_replay_corrupt_tail() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/corrupt_tail.jsonl")

    pending_store, _, _ = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        config=cfg,
    )

    # Corrupt tail discarded, valid first entry recovered
    assert len(pending_store) == 1


def test_replay_corrupt_middle() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/corrupt_middle.jsonl")

    with pytest.raises(ValueError) as exc_info:
        replay_log(
            active_path=wal_path,
            rotated_path=None,
            snapshot_state_version=0,
            config=cfg,
        )
    assert "WAL corrupt" in str(exc_info.value)
