"""Tests for snapshot persistence, restore, and compatibility validation (T19 & T20)."""

import time
from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config, to_static_config
from azuma_router.errors import IncompatibleSnapshotError
from azuma_router.quota import QuotaLedger
from azuma_router.snapshot import restore_snapshot, write_snapshot
from azuma_router.state import init_state
from azuma_router.types import SnapshotMeta


def test_snapshot_write_atomic_ac22(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)
    ledger = QuotaLedger(cfg)
    static = to_static_config(cfg)

    meta = SnapshotMeta(
        schema_version=1,
        state_version=42,
        featurizer_version=1,
        arm_ids=[arm.arm_id for arm in cfg.arms],
        static=static,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="hash123",
    )

    path, n_bytes = write_snapshot(tmp_path, state, ledger, meta)
    assert path == tmp_path / "snapshot"
    assert n_bytes > 0

    assert (tmp_path / "snapshot" / "state.safetensors").exists()
    assert (tmp_path / "snapshot" / "state.json").exists()
    assert not (tmp_path / "snapshot.tmp").exists()


def test_restore_roundtrip_bitwise_ac24(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)
    ledger = QuotaLedger(cfg)
    static = to_static_config(cfg)

    arm_id = cfg.arms[0].arm_id
    ledger.record(arm_id, 10.0, time.time())

    meta = SnapshotMeta(
        schema_version=1,
        state_version=50,
        featurizer_version=1,
        arm_ids=[arm.arm_id for arm in cfg.arms],
        static=static,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="hash123",
    )

    write_snapshot(tmp_path, state, ledger, meta)

    restored_state, restored_ledger, restored_meta = restore_snapshot(tmp_path, cfg)
    assert restored_state is not None

    np.testing.assert_array_equal(np.array(restored_state.A), np.array(state.A))
    np.testing.assert_array_equal(np.array(restored_state.b), np.array(state.b))
    np.testing.assert_array_equal(np.array(restored_state.n_pulls), np.array(state.n_pulls))
    np.testing.assert_array_equal(np.array(restored_state.mask), np.array(state.mask))
    np.testing.assert_array_equal(np.array(restored_state.rng_key), np.array(state.rng_key))

    assert restored_meta.state_version == 50

    now = time.time()
    np.testing.assert_array_equal(restored_ledger.utilization(now), ledger.utilization(now))


def test_restore_fail_fast_ac23(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)
    ledger = QuotaLedger(cfg)
    static = to_static_config(cfg)

    meta = SnapshotMeta(
        schema_version=1,
        state_version=1,
        featurizer_version=1,
        arm_ids=["different_arm@provider"],  # Incompatible arm_ids
        static=static,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="hash123",
    )

    write_snapshot(tmp_path, state, ledger, meta)

    with pytest.raises(IncompatibleSnapshotError) as exc_info:
        restore_snapshot(tmp_path, cfg)
    assert "arm_ids" in str(exc_info.value)


def test_restore_snapshot_old_recovery(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)
    ledger = QuotaLedger(cfg)
    static = to_static_config(cfg)

    meta = SnapshotMeta(
        schema_version=1,
        state_version=99,
        featurizer_version=1,
        arm_ids=[arm.arm_id for arm in cfg.arms],
        static=static,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="hash123",
    )

    write_snapshot(tmp_path, state, ledger, meta)

    # Rename snapshot -> snapshot.old to simulate crash between renames
    snap_dir = tmp_path / "snapshot"
    snap_old_dir = tmp_path / "snapshot.old"
    snap_dir.rename(snap_old_dir)

    restored_state, _restored_ledger, restored_meta = restore_snapshot(tmp_path, cfg)
    assert restored_state is not None
    assert restored_meta is not None
    assert restored_meta.state_version == 99


def test_restore_config_sha256_mismatch_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)
    ledger = QuotaLedger(cfg)
    static = to_static_config(cfg)

    meta = SnapshotMeta(
        schema_version=1,
        state_version=10,
        featurizer_version=1,
        arm_ids=[arm.arm_id for arm in cfg.arms],
        static=static,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger=ledger.dump(),
        config_sha256="different_sha256_hash",
    )

    write_snapshot(tmp_path, state, ledger, meta)

    restored_state, _, restored_meta = restore_snapshot(tmp_path, cfg)
    assert restored_state is not None
    assert restored_meta.state_version == 10

    captured = capsys.readouterr()
    assert "WARNING" in captured.err or "Config SHA256 mismatch" in captured.err
