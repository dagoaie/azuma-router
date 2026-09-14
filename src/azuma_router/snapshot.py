"""Atomic snapshot persistence and fail-fast restoration.

Complies with contract §3.7, §4.3, FR-25..FR-28, ADR-0009, ADR-0017.
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from safetensors.numpy import load_file, save_file

from azuma_router.config import Config
from azuma_router.errors import IncompatibleSnapshotError
from azuma_router.quota import QuotaLedger
from azuma_router.state import state_from_numpy, state_to_numpy
from azuma_router.types import BanditState, SnapshotMeta


def compute_config_sha256(config_path_or_str: str | Path | Config) -> str:
    """Computes SHA-256 hash of configuration."""
    if isinstance(config_path_or_str, Config):
        # Hash string representation of arms and bandit hyperparams
        content = json.dumps(
            {
                "arms": [arm.arm_id for arm in config_path_or_str.arms],
                "bandit": config_path_or_str.bandit.seed,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    path = Path(config_path_or_str)
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def write_snapshot(
    dir_path: str | Path,
    state: BanditState,
    ledger: QuotaLedger,
    meta: SnapshotMeta,
) -> tuple[Path, int]:
    """Writes state and ledger to snapshot directory via atomic rename (FR-25).

    Returns:
        (path_to_snapshot, total_bytes_written)
    """
    root_dir = Path(dir_path)
    root_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = root_dir / "snapshot.tmp"
    target_dir = root_dir / "snapshot"
    old_dir = root_dir / "snapshot.old"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. state.safetensors
    tensor_dict = state_to_numpy(state)

    tensor_path = tmp_dir / "state.safetensors"
    save_file(tensor_dict, str(tensor_path))

    # 2. state.json
    json_path = tmp_dir / "state.json"
    json_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")

    # Calculate written bytes
    total_bytes = tensor_path.stat().st_size + json_path.stat().st_size

    # fsync files and tmp directory
    with tensor_path.open("rb") as f:
        os.fsync(f.fileno())
    with json_path.open("rb") as f:
        os.fsync(f.fileno())

    tmp_dir_fd = os.open(str(tmp_dir), os.O_RDONLY)
    try:
        os.fsync(tmp_dir_fd)
    finally:
        os.close(tmp_dir_fd)

    # Atomic directory rename dance (ADR-0009)
    if old_dir.exists():
        shutil.rmtree(old_dir)

    if target_dir.exists():
        target_dir.rename(old_dir)

    tmp_dir.rename(target_dir)

    if old_dir.exists():
        shutil.rmtree(old_dir)

    return target_dir, total_bytes


def _validate_snapshot_compatibility(meta: SnapshotMeta, config: Config) -> None:
    """Performs fail-fast shape and version checks against running Config (FR-26)."""
    expected_arm_ids = [arm.arm_id for arm in config.arms]
    if meta.schema_version != 1:
        raise IncompatibleSnapshotError(f"Unsupported schema_version: {meta.schema_version}")
    if meta.featurizer_version != 1:
        msg = f"Unsupported featurizer_version: {meta.featurizer_version}"
        raise IncompatibleSnapshotError(msg)
    if meta.static.d != config.bandit.d:
        msg = f"d mismatch: snapshot={meta.static.d}, config={config.bandit.d}"
        raise IncompatibleSnapshotError(msg)
    if meta.static.k_max != config.bandit.k_max:
        msg = f"k_max mismatch: snapshot={meta.static.k_max}, config={config.bandit.k_max}"
        raise IncompatibleSnapshotError(msg)
    if meta.static.w_max != config.bandit.w_max:
        msg = f"w_max mismatch: snapshot={meta.static.w_max}, config={config.bandit.w_max}"
        raise IncompatibleSnapshotError(msg)
    if meta.arm_ids != expected_arm_ids:
        msg = f"arm_ids mismatch: snapshot={meta.arm_ids}, config={expected_arm_ids}"
        raise IncompatibleSnapshotError(msg)


def restore_snapshot(
    dir_path: str | Path,
    config: Config,
) -> tuple[BanditState | None, QuotaLedger | None, SnapshotMeta | None]:
    """Restores bandit state and ledger with fail-fast compatibility check (FR-26)."""
    root_dir = Path(dir_path)
    tmp_dir = root_dir / "snapshot.tmp"
    target_dir = root_dir / "snapshot"
    old_dir = root_dir / "snapshot.old"

    # Cleanup residual tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Crash recovery: restore old if target does not exist
    if old_dir.exists() and not target_dir.exists():
        old_dir.rename(target_dir)

    if not target_dir.exists():
        return None, None, None

    json_path = target_dir / "state.json"
    tensor_path = target_dir / "state.safetensors"

    if not json_path.exists() or not tensor_path.exists():
        msg = f"Snapshot directory {target_dir} is missing state.json or state.safetensors"
        raise IncompatibleSnapshotError(msg)

    # Read and parse metadata
    try:
        meta_dict = json.loads(json_path.read_text(encoding="utf-8"))
        meta = SnapshotMeta.from_dict(meta_dict)
    except Exception as err:
        msg = f"Failed to parse snapshot metadata: {err}"
        raise IncompatibleSnapshotError(msg) from err

    # Fail-fast compatibility checks (FR-26)
    _validate_snapshot_compatibility(meta, config)

    current_sha256 = compute_config_sha256(config)
    if meta.config_sha256 != current_sha256:
        warn_payload = {
            "ts": time.time(),
            "level": "warning",
            "event": "config_sha256_mismatch",
            "snapshot_sha256": meta.config_sha256,
            "current_sha256": current_sha256,
        }
        sys.stderr.write(json.dumps(warn_payload, allow_nan=False) + "\n")
        sys.stderr.flush()

    # Load tensors
    try:
        tensors = load_file(str(tensor_path))
    except Exception as err:
        msg = f"Failed to load safetensors file {tensor_path}: {err}"
        raise IncompatibleSnapshotError(msg) from err

    state = state_from_numpy(tensors)

    ledger = QuotaLedger.load(meta.ledger, config)
    return state, ledger, meta


# Public API aliases (ADR-0012)
snapshot = write_snapshot
restore = restore_snapshot
