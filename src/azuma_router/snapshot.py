"""Atomic snapshot persistence and fail-fast restoration.

Complies with contract §3.7, §4.3, FR-25..FR-28, ADR-0009, ADR-0017.
"""

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from azuma_router.config import Config
from azuma_router.errors import AzumaError, IncompatibleSnapshotError, SnapshotWriteError
from azuma_router.quota import QuotaLedger
from azuma_router.state import state_from_numpy, state_to_numpy
from azuma_router.types import BanditState, SnapshotMeta
from azuma_router.wal import (
    fsync_directory,
    make_private_dir,
    make_private_file,
    require_trusted_paths,
)

_META_FILE = "state.json"
_TENSOR_FILE = "state.safetensors"


def compute_config_sha256(config: Config) -> str:
    """SHA-256 of the canonical JSON form of the whole validated config (P3-SF-4).

    Hashing the dataclass rather than the TOML bytes keeps the hash stable across
    comments and key order while still flagging any hyperparameter or arm change.
    """
    canonical = json.dumps(dataclasses.asdict(config), sort_keys=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    try:
        return _write_snapshot_files(Path(dir_path), state, ledger, meta)
    except AzumaError:
        raise
    except Exception as err:
        # Any backend failure (OSError, safetensors, serialization) is one fail-stop policy:
        # the daemon must not keep serving without snapshots (P6-RR-1).
        raise SnapshotWriteError(f"snapshot write failed: {err}") from err


def _write_snapshot_files(
    root_dir: Path,
    state: BanditState,
    ledger: QuotaLedger,
    meta: SnapshotMeta,
) -> tuple[Path, int]:
    make_private_dir(root_dir)

    tmp_dir = root_dir / "snapshot.tmp"
    target_dir = root_dir / "snapshot"
    old_dir = root_dir / "snapshot.old"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    make_private_dir(tmp_dir)

    # 1. state.safetensors
    tensor_dict = state_to_numpy(state)

    tensor_path = tmp_dir / _TENSOR_FILE
    save_file(tensor_dict, str(tensor_path))
    make_private_file(tensor_path)

    # 2. state.json
    json_path = tmp_dir / _META_FILE
    json_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    make_private_file(json_path)

    # Calculate written bytes
    total_bytes = tensor_path.stat().st_size + json_path.stat().st_size

    # fsync files and tmp directory
    with tensor_path.open("rb") as f:
        os.fsync(f.fileno())
    with json_path.open("rb") as f:
        os.fsync(f.fileno())

    fsync_directory(tmp_dir)

    # Atomic directory rename dance (ADR-0009)
    if old_dir.exists():
        shutil.rmtree(old_dir)

    if target_dir.exists():
        target_dir.rename(old_dir)

    tmp_dir.rename(target_dir)
    fsync_directory(root_dir)  # the new snapshot/ entry is durable before old is dropped

    if old_dir.exists():
        shutil.rmtree(old_dir)

    return target_dir, total_bytes


def _validate_snapshot_compatibility(meta: SnapshotMeta, config: Config) -> None:
    """Performs fail-fast shape and version checks against running Config (FR-26)."""
    expected_arm_ids = [arm.arm_id for arm in config.arms]
    if meta.schema_version != 2:
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
    if type(meta.state_version) is not int or meta.state_version < 0:
        raise IncompatibleSnapshotError("state_version must be a non-negative integer")
    if type(meta.selection_sequence) is not int or meta.selection_sequence < meta.state_version:
        raise IncompatibleSnapshotError("selection_sequence must be >= state_version")


def _validate_tensor_schema(tensors: dict[str, np.ndarray], config: Config) -> None:
    """Rejects a safetensors payload whose keys, shapes or dtypes are not a BanditState."""
    k_max = config.bandit.k_max
    d = config.bandit.d
    schemas = {
        "A": ((k_max, d, d), np.dtype(np.float64)),
        "b": ((k_max, d), np.dtype(np.float64)),
        "n_pulls": ((k_max,), np.dtype(np.int32)),
        "mask": ((k_max,), np.dtype(bool)),
        "rng_key": ((2,), np.dtype(np.uint32)),
    }
    if set(tensors) != set(schemas):
        raise IncompatibleSnapshotError(
            f"Snapshot tensor keys must be {sorted(schemas)}, got {sorted(tensors)}"
        )
    for name, (shape, dtype) in schemas.items():
        value = tensors[name]
        if value.shape != shape or value.dtype != dtype:
            raise IncompatibleSnapshotError(
                f"Invalid snapshot tensor {name}: expected {dtype}{shape}, "
                f"got {value.dtype}{value.shape}"
            )


def _validate_state_tensors(tensors: dict[str, np.ndarray], config: Config) -> None:
    """Rejects malformed or numerically invalid persisted bandit state before JAX conversion."""
    _validate_tensor_schema(tensors, config)
    a = tensors["A"]
    b = tensors["b"]
    n_pulls = tensors["n_pulls"]
    mask = tensors["mask"]
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise IncompatibleSnapshotError("Snapshot tensors A and b must be finite")
    if not np.array_equal(a, np.swapaxes(a, -1, -2)):
        raise IncompatibleSnapshotError("Snapshot tensor A must be symmetric")
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as err:
        raise IncompatibleSnapshotError("Snapshot tensor A must be positive definite") from err
    if np.any(n_pulls < 0):
        raise IncompatibleSnapshotError("Snapshot tensor n_pulls must be non-negative")
    active = len(config.arms)
    expected_mask = np.array([True] * active + [False] * (config.bandit.k_max - active))
    if not np.array_equal(mask, expected_mask):
        raise IncompatibleSnapshotError("Snapshot tensor mask is not canonical for configured arms")


def restore_snapshot(
    dir_path: str | Path,
    config: Config,
) -> tuple[BanditState | None, QuotaLedger | None, SnapshotMeta | None]:
    """Restores bandit state and ledger with fail-fast compatibility check (FR-26)."""
    root_dir = Path(dir_path)
    tmp_dir = root_dir / "snapshot.tmp"
    target_dir = root_dir / "snapshot"
    old_dir = root_dir / "snapshot.old"

    # Trust nothing this process is about to delete, rename or read (FR-25b)
    require_trusted_paths(
        [
            root_dir,
            tmp_dir,
            old_dir,
            target_dir,
            *(
                directory / name
                for directory in (old_dir, target_dir)
                for name in (_META_FILE, _TENSOR_FILE)
            ),
        ]
    )

    # Cleanup residual tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # Crash recovery: restore old if target does not exist
    if old_dir.exists() and not target_dir.exists():
        old_dir.rename(target_dir)

    if not target_dir.exists():
        return None, None, None

    json_path = target_dir / _META_FILE
    tensor_path = target_dir / _TENSOR_FILE

    if not json_path.exists() or not tensor_path.exists():
        msg = f"Snapshot directory {target_dir} is missing state.json or state.safetensors"
        raise IncompatibleSnapshotError(msg)

    # Read (OSError propagates: unreadable != incompatible) and parse metadata
    meta_text = json_path.read_text(encoding="utf-8")
    try:
        meta = SnapshotMeta.from_dict(json.loads(meta_text))
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

    # A snapshot that passed the version/shape checks can still carry tensors or a ledger this
    # build cannot decode; that is an operator problem, not a traceback (P5-MF-3).
    try:
        _validate_state_tensors(tensors, config)
        state = state_from_numpy(tensors)
        ledger = QuotaLedger.load(meta.ledger, config)
    except Exception as err:
        msg = f"Failed to decode snapshot {dir_path}: {err!r}"
        raise IncompatibleSnapshotError(msg) from err
    return state, ledger, meta


# Public API aliases (ADR-0012)
snapshot = write_snapshot
restore = restore_snapshot
