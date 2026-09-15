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


def test_replay_corrupt_tail(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = tmp_path / "corrupt_tail.jsonl"
    valid_select = Path("fixtures/wal/pending_only.jsonl").read_text(encoding="utf-8")
    wal_path.write_text(
        valid_select + '{"phase":"select","decision_id":"truncated"', encoding="utf-8"
    )

    pending_store, _, _, _rng_key, _selection_sequence = replay_log(
        active_path=wal_path,
        rotated_path=None,
        snapshot_state_version=0,
        snapshot_selection_sequence=0,
        config=cfg,
    )

    # Corrupt tail discarded, valid first entry recovered
    assert len(pending_store) == 1


def test_replay_rejects_a_complete_corrupt_final_line(tmp_path: Path) -> None:
    """Only an unterminated final line can be a crash truncation artefact (FR-23)."""
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = tmp_path / "complete_corrupt_tail.jsonl"
    wal_path.write_text('{"phase":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="WAL corrupt"):
        replay_log(
            active_path=wal_path,
            rotated_path=None,
            snapshot_state_version=0,
            snapshot_selection_sequence=0,
            config=cfg,
        )


def test_replay_rejects_a_second_active_carried_select(tmp_path: Path) -> None:
    """Rotation emits one carried copy per live decision, never an unbounded duplicate stream."""
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    entry = json.loads(Path("fixtures/wal/pending_only.jsonl").read_text(encoding="utf-8"))
    rotated_path = tmp_path / "decisions.1.jsonl"
    active_path = tmp_path / "decisions.jsonl"
    rotated_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    active_path.write_text(json.dumps(entry) + "\n" + json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts"):
        replay_log(
            active_path=active_path,
            rotated_path=rotated_path,
            snapshot_state_version=0,
            snapshot_selection_sequence=1,
            config=cfg,
        )


def test_replay_corrupt_middle(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    wal_path = tmp_path / "corrupt_middle.jsonl"
    valid_select = Path("fixtures/wal/pending_only.jsonl").read_text(encoding="utf-8")
    wal_path.write_text(valid_select + '{"phase":\n' + valid_select, encoding="utf-8")

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
    line = json.loads(Path("fixtures/wal/pending_only.jsonl").read_text(encoding="utf-8"))
    # Notice no ledger_event_id in line!
    line.pop("ledger_event_id")
    wal_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid schema"):
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
