"""Tests for WAL replay (T18 & AC-20)."""

import json
from pathlib import Path

import pytest

from azuma_router.config import load_config
from azuma_router.errors import JournalWriteError
from azuma_router.wal import DecisionLog, replay_log


def test_replay_pending_only_ac20() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/pending_only.jsonl")

    pending_store, applied_reports, _applied_ids, _rng_key, _selection_sequence = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        snapshot_selection_sequence=0,
        config=cfg,
    )

    assert len(pending_store) == 1
    assert "018f6c2e-0000-7000-8000-000000000001" in pending_store
    assert len(applied_reports) == 0


def test_replay_reported_ac20() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/reported.jsonl")

    pending_store, applied_reports, applied_ids, _rng_key, _selection_sequence = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        snapshot_selection_sequence=0,
        config=cfg,
    )

    assert len(pending_store) == 0
    assert len(applied_reports) == 1
    assert applied_reports[0][0] == 1  # state_version 1
    assert "018f6c2e-0000-7000-8000-000000000001" in applied_ids


def test_replay_corrupt_tail() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = Path("fixtures/wal/corrupt_tail.jsonl")

    pending_store, _, _, _rng_key, _selection_sequence = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        snapshot_selection_sequence=0,
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
            snapshot_selection_sequence=0,
            config=cfg,
        )
    assert "WAL corrupt" in str(exc_info.value)


def test_replay_select_missing_ledger_event_id_raises(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = tmp_path / "missing_eid.jsonl"
    line = {
        "phase": "select",
        "decision_id": "018f6c2e-0000-7000-8000-000000000001",
        "ts": 1757800000.0,
        "state_version": 0,
        "arm_index": 0,
        "arm_id": "gpt-5-codex@openai-codex",
        "propensity": 0.8123,
        "x": [1.0] * 32,
        "policy": [0.8123] + [0.0] * 31,
        "score": [0.71] + [-1e9] * 31,
        "exploit": [0.5] + [0.0] * 31,
        "bonus": [0.21] + [0.0] * 31,
        "shadow_price": [0.0] * 32,
        "utilization": {"gpt-5-codex@openai-codex": [0.0, 0.0]},
        "est_units": 1.0,
        "featurizer_version": 1,
    }
    # Notice no ledger_event_id in line!
    wal_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ledger_event_id"):
        replay_log(
            active_path=wal_path,
            rotated_path=None,
            snapshot_state_version=0,
            snapshot_selection_sequence=0,
            config=cfg,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_sequence", 2),
        ("selection_sequence", True),
        ("selection_sequence", 1.5),
        ("rng_key", [1]),
        ("rng_key", [1, -1]),
        ("rng_key", [1, 2**32]),
        ("rng_key", [1, True]),
    ],
)
def test_replay_rejects_invalid_rng_transition(tmp_path: Path, field: str, value: object) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    entry = json.loads(Path("fixtures/wal/pending_only.jsonl").read_text(encoding="utf-8"))
    entry[field] = value
    wal_path = tmp_path / "invalid_rng.jsonl"
    wal_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"RNG transition|selection_sequence"):
        replay_log(
            active_path=wal_path,
            rotated_path=None,
            snapshot_state_version=0,
            snapshot_selection_sequence=0,
            config=cfg,
        )


def test_decision_log_rejects_a_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("must remain untouched", encoding="utf-8")
    wal_path = tmp_path / "decisions.jsonl"
    wal_path.symlink_to(target)

    with pytest.raises(JournalWriteError, match="WAL open failed"):
        DecisionLog(wal_path)
    assert target.read_text(encoding="utf-8") == "must remain untouched"
