"""Ownership/mode verification of persisted state before it is trusted (audit #2, FR-25b)."""

import os
import stat
from pathlib import Path

import pytest

from azuma_router.config import load_config, to_static_config
from azuma_router.errors import UnsafeStateError
from azuma_router.quota import QuotaLedger
from azuma_router.snapshot import restore_snapshot, write_snapshot
from azuma_router.state import init_state
from azuma_router.types import SnapshotMeta
from azuma_router.wal import require_trusted_paths


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _write_valid_snapshot(root: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    meta = SnapshotMeta(
        schema_version=2,
        state_version=1,
        featurizer_version=1,
        arm_ids=[arm.arm_id for arm in cfg.arms],
        static=to_static_config(cfg),
        seed=1,
        selection_sequence=1,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="hash",
    )
    write_snapshot(root, init_state(cfg), ledger, meta)


class TestRequireTrustedPaths:
    def test_owner_only_paths_pass(self, tmp_path: Path) -> None:
        directory = tmp_path / "dir"
        directory.mkdir(mode=0o700)
        journal = tmp_path / "decisions.jsonl"
        journal.write_text("")
        journal.chmod(0o600)
        require_trusted_paths([directory, journal])

    def test_missing_paths_are_skipped(self, tmp_path: Path) -> None:
        require_trusted_paths([tmp_path / "absent"])

    def test_world_readable_but_not_writable_files_pass(self, tmp_path: Path) -> None:
        journal = tmp_path / "decisions.jsonl"
        journal.write_text("")
        journal.chmod(0o644)
        require_trusted_paths([journal])

    @pytest.mark.parametrize("mode", [0o770, 0o707, 0o777, 0o720])
    def test_group_or_world_writable_directory_is_refused(self, tmp_path: Path, mode: int) -> None:
        directory = tmp_path / "dir"
        directory.mkdir()
        directory.chmod(mode)
        with pytest.raises(UnsafeStateError, match=str(directory)):
            require_trusted_paths([directory])

    @pytest.mark.parametrize("mode", [0o660, 0o606, 0o666])
    def test_group_or_world_writable_file_is_refused(self, tmp_path: Path, mode: int) -> None:
        journal = tmp_path / "decisions.jsonl"
        journal.write_text("")
        journal.chmod(mode)
        with pytest.raises(UnsafeStateError, match="writable"):
            require_trusted_paths([journal])

    def test_foreign_owner_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        journal = tmp_path / "decisions.jsonl"
        journal.write_text("")
        monkeypatch.setattr(os, "geteuid", lambda: journal.stat().st_uid + 1)
        with pytest.raises(UnsafeStateError, match="owned by uid"):
            require_trusted_paths([journal])

    def test_checking_never_alters_permissions(self, tmp_path: Path) -> None:
        directory = tmp_path / "dir"
        directory.mkdir()
        directory.chmod(0o777)
        with pytest.raises(UnsafeStateError):
            require_trusted_paths([directory])
        assert _mode(directory) == 0o777


class TestRestoreRefusesUntrustedSnapshot:
    def test_a_private_snapshot_still_restores(self, tmp_path: Path) -> None:
        root = tmp_path / ".azuma"
        _write_valid_snapshot(root)
        cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
        state, _ledger, meta = restore_snapshot(root, cfg)
        assert state is not None
        assert meta is not None

    @pytest.mark.parametrize("victim", ["", "snapshot", "snapshot/state.json"])
    def test_group_writable_snapshot_artifact_is_refused(self, tmp_path: Path, victim: str) -> None:
        root = tmp_path / ".azuma"
        _write_valid_snapshot(root)
        target = root / victim
        target.chmod(0o770 if target.is_dir() else 0o660)
        cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
        with pytest.raises(UnsafeStateError):
            restore_snapshot(root, cfg)

    def test_refusal_happens_before_any_cleanup_or_rename(self, tmp_path: Path) -> None:
        root = tmp_path / ".azuma"
        _write_valid_snapshot(root)
        leftover = root / "snapshot.tmp"
        leftover.mkdir()
        (leftover / "marker").write_text("x")
        (root / "snapshot").chmod(0o770)
        cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
        with pytest.raises(UnsafeStateError):
            restore_snapshot(root, cfg)
        assert (leftover / "marker").exists()
        assert _mode(root / "snapshot") == 0o770

    def test_untrusted_snapshot_old_is_refused_before_crash_recovery_rename(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / ".azuma"
        _write_valid_snapshot(root)
        (root / "snapshot").rename(root / "snapshot.old")
        (root / "snapshot.old").chmod(0o777)
        cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
        with pytest.raises(UnsafeStateError):
            restore_snapshot(root, cfg)
        assert not (root / "snapshot").exists()
