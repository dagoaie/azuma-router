"""Single-thread JSON-RPC 2.0 daemon over stdio.

Complies with contract §3.8, §5, FR-29..FR-33, ADR-0010, ADR-0013, ADR-0014.
"""

import argparse
import datetime
import json
import math
import secrets
import signal
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from azuma_router.config import Config, load_config, to_static_config
from azuma_router.errors import (
    ConfigError,
    InvalidContextError,
    JournalWriteError,
    NumericalIntegrityError,
    RpcError,
)
from azuma_router.featurizer import FEATURIZER_VERSION, featurize
from azuma_router.linucb import _compile_count, select_jit, update_jit
from azuma_router.quota import QuotaLedger, calc_est_units
from azuma_router.reward import compute_reward
from azuma_router.snapshot import (
    compute_config_sha256,
    restore_snapshot,
    write_snapshot,
)
from azuma_router.state import (
    init_state,
    make_quota_view,
    make_warmup_inputs,
    to_jax_array,
)
from azuma_router.types import (
    MAX_INT31,
    BanditState,
    Pending,
    RoutingRequest,
    SnapshotMeta,
)
from azuma_router.wal import DecisionLog, PendingStore, replay_log


def uuid7() -> str:
    """Generates a UUIDv7 string (48-bit timestamp ms + random)."""
    ts_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    high = (ts_ms << 16) | (0x7 << 12) | rand_a
    low = (0x2 << 62) | rand_b
    return str(uuid.UUID(int=(high << 64) | low))


def _log_event(
    stream: TextIO,
    level: str,
    event: str,
    timestamp: float | None = None,
    **fields: Any,
) -> None:
    now = timestamp if timestamp is not None else time.time()
    payload = {
        "ts": now,
        "level": level,
        "event": event,
        **fields,
    }
    stream.write(json.dumps(payload, allow_nan=False) + "\n")
    stream.flush()


def _check_finite_float(
    val: Any,
    field_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        raise RpcError(-32602, f"{field_name} must be numeric", data={"field": field_name})
    fval = float(val)
    if not math.isfinite(fval):
        raise RpcError(-32602, f"{field_name} must be finite", data={"field": field_name})
    if min_val is not None and fval < min_val:
        raise RpcError(-32602, f"{field_name} must be >= {min_val}", data={"field": field_name})
    if max_val is not None and fval > max_val:
        raise RpcError(-32602, f"{field_name} must be <= {max_val}", data={"field": field_name})
    return fval


def _check_non_neg_int(val: Any, field_name: str) -> int:
    msg = f"{field_name} must be int between 0 and 2**31-1"
    if val is None or isinstance(val, bool) or not isinstance(val, int):
        raise RpcError(-32602, msg, data={"field": field_name})
    if not (0 <= val <= MAX_INT31):
        raise RpcError(-32602, msg, data={"field": field_name})
    return val


def _build_allowed_mask(allowed_list: Any, config: Config) -> list[bool]:
    k_max = config.bandit.k_max
    allowed_mask = [False] * k_max
    all_arm_ids = [arm.arm_id for arm in config.arms]

    if allowed_list is not None:
        if not isinstance(allowed_list, list):
            raise RpcError(-32602, "allowed_arms must be a list", data={"field": "allowed_arms"})
        for item in allowed_list:
            if not isinstance(item, str) or item not in all_arm_ids:
                raise RpcError(
                    -32602,
                    f"Unknown arm in allowed_arms: {item!r}",
                    data={"field": f"allowed_arms[{item}]"},
                )
        for i, arm_id in enumerate(all_arm_ids):
            allowed_mask[i] = arm_id in allowed_list
    else:
        for i in range(len(all_arm_ids)):
            allowed_mask[i] = True

    return allowed_mask


def _build_select_diagnostics(
    arms: tuple[Any, ...],
    diag: Any,
    util_matrix: np.ndarray,
    est_units_arr: np.ndarray,
) -> dict[str, Any]:
    policies = [float(diag.policy[i]) for i in range(len(arms))]

    def _val_or_none(val: Any, active: bool) -> float | None:
        return float(val) if active else None

    return {
        "score": {
            arm.arm_id: _val_or_none(diag.score[i], policies[i] > 0) for i, arm in enumerate(arms)
        },
        "exploit": {
            arm.arm_id: _val_or_none(diag.exploit[i], policies[i] > 0) for i, arm in enumerate(arms)
        },
        "bonus": {
            arm.arm_id: _val_or_none(diag.bonus[i], policies[i] > 0) for i, arm in enumerate(arms)
        },
        "shadow_price": {
            arm.arm_id: _val_or_none(diag.shadow_price[i], policies[i] > 0)
            for i, arm in enumerate(arms)
        },
        "policy": {arm.arm_id: policies[i] for i, arm in enumerate(arms)},
        "utilization": {
            arm.arm_id: {
                f"{w.kind}:{w.duration_s or w.period}": float(util_matrix[i, j])
                for j, w in enumerate(arm.windows)
            }
            for i, arm in enumerate(arms)
        },
        "est_units": {arm.arm_id: float(est_units_arr[i]) for i, arm in enumerate(arms)},
    }


class Router:
    """Orchestrates bandit state, quota ledger, WAL, and snapshot lifecycle (ADR-0010)."""

    def __init__(
        self,
        config: Config,
        snapshot_dir: str | Path | None = None,
        decision_log_path: str | Path | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.static = to_static_config(config)
        self.clock = clock_fn or time.time

        s_dir = snapshot_dir or config.bandit.snapshot_dir
        self.snapshot_dir = Path(s_dir)

        d_log = decision_log_path or config.bandit.decision_log_path
        self.decision_log_path = Path(d_log)

        self.ready = False
        self.state_version = 0
        self.updates_since_snapshot = 0
        self.last_snapshot_ts = self.clock()
        self.last_ts = self.clock()

        self.state: BanditState
        self.ledger: QuotaLedger
        self.pending_store = PendingStore()
        self.dlog: DecisionLog | None = None

    def boot(self) -> dict[str, Any]:
        """Initializes or restores state, replays WAL, warms up JIT, and marks ready (FR-30)."""
        # 1. Restore snapshot or clean init
        res_snap = restore_snapshot(self.snapshot_dir, self.config)
        restored_state, restored_ledger, restored_meta = res_snap

        if restored_state is not None and restored_ledger is not None and restored_meta is not None:
            self.state = restored_state
            self.ledger = restored_ledger
            self.state_version = restored_meta.state_version
            restored_from_snapshot = True
        else:
            # Check inconsistent state: active WAL exists with state_version > 0 but no snapshot
            if self.decision_log_path.exists() and not (self.snapshot_dir / "snapshot").exists():
                with self.decision_log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if '"state_version":' in line:
                            try:
                                data = json.loads(line)
                                if data.get("state_version", 0) > 0:
                                    msg = (
                                        "Active WAL exists with state_version > 0 "
                                        "but no snapshot/ directory"
                                    )
                                    _log_event(sys.stderr, "error", "wal_inconsistent", error=msg)
                                    sys.exit(3)
                            except Exception:
                                pass

            self.state = init_state(self.config)
            self.ledger = QuotaLedger(self.config)
            self.state_version = 0
            restored_from_snapshot = False

        # 2. Replay WAL
        r_name = f"{self.decision_log_path.stem}.1{self.decision_log_path.suffix}"
        rotated_path = self.decision_log_path.with_name(r_name)
        try:
            p_store, applied_reports, _applied_ids = replay_log(
                active_path=self.decision_log_path,
                rotated_path=rotated_path if rotated_path.exists() else None,
                snapshot_state_version=self.state_version,
                config=self.config,
                ledger=self.ledger,
            )
        except ValueError as err:
            _log_event(sys.stderr, "error", "replay_failed", error=str(err))
            sys.exit(3)

        self.pending_store = p_store

        # Re-apply reports subsequent to snapshot
        for s_ver, x_arr, arm_idx, reward in applied_reports:
            self.state = update_jit(
                self.state,
                to_jax_array(x_arr),
                arm_idx,
                reward,
                self.static,
            )
            self.state_version = max(self.state_version, s_ver)

        # 3. Open DecisionLog
        self.dlog = DecisionLog(self.decision_log_path, self.config.bandit.decision_log_max_bytes)

        # 4. Warm-up JIT: triggers both compilations; results are discarded so the
        # dummy pull never touches n_pulls or rng_key (D16).
        t_start = time.perf_counter()
        dummy_x, dummy_qv = make_warmup_inputs(self.static)

        warm_state, _, _, _ = select_jit(self.state, dummy_x, dummy_qv, self.static)
        warm_state = update_jit(warm_state, dummy_x, 0, 0.0, self.static)
        warm_state.n_pulls.block_until_ready()  # measure real compile time, not dispatch
        warmup_ms = (time.perf_counter() - t_start) * 1000.0

        self.ready = True

        _log_event(
            sys.stderr,
            "info",
            "ready",
            state_version=self.state_version,
            warmup_ms=warmup_ms,
        )

        return {
            "state_version": self.state_version,
            "arms": [arm.arm_id for arm in self.config.arms],
            "restored_from_snapshot": restored_from_snapshot,
            "pending_replayed": len(self.pending_store),
            "warmup_ms": warmup_ms,
        }

    def _check_clock_skew(self, now: float) -> None:
        if now < self.last_ts - 1.0:
            _log_event(
                sys.stderr,
                "warning",
                "clock_skew",
                timestamp=now,
                delta_s=now - self.last_ts,
            )
        self.last_ts = now

    def _sweep_expired(self) -> None:
        """Sweeps expired pending decisions and writes expire WAL lines."""
        now = self.clock()
        self._check_clock_skew(now)
        expired = self.pending_store.sweep(now, self.config.bandit.pending_ttl_s)
        if expired:
            _log_event(
                sys.stderr,
                "info",
                "pending_expired",
                timestamp=now,
                count=len(expired),
                decision_ids=[p.decision_id for p in expired],
            )
        for p in expired:
            if self.dlog is not None:
                self.dlog.append(
                    {
                        "phase": "expire",
                        "decision_id": p.decision_id,
                        "ts": now,
                        "state_version": self.state_version,
                        "ts_select": p.ts_select,
                    }
                )

    def select(self, params: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates select step (FR-20, FR-31)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        self._sweep_expired()
        now = self.clock()

        # Parse request
        req_raw = params.get("request")
        if not isinstance(req_raw, dict):
            raise RpcError(-32602, "Missing or invalid 'request' object", data={"field": "request"})

        try:
            req = RoutingRequest(
                task_kind=req_raw.get("task_kind"),  # type: ignore[arg-type]
                input_tokens=req_raw.get("input_tokens", -1),
                expected_output_tokens=req_raw.get("expected_output_tokens"),
                has_code=req_raw.get("has_code", False),
                has_tools=req_raw.get("has_tools", False),
                has_images=req_raw.get("has_images", False),
                agent_depth=req_raw.get("agent_depth", 0),
                diff_lines=req_raw.get("diff_lines"),
                priority=req_raw.get("priority", "normal"),
            )
            x_np = featurize(req)
        except InvalidContextError as exc:
            err_data = {"field": exc.field, "reason": exc.reason}
            raise RpcError(-32001, "Invalid context", data=err_data) from exc

        # Allowed arms mask
        k_max = self.static.k_max
        allowed_mask = _build_allowed_mask(params.get("allowed_arms"), self.config)

        # est_units
        est_units_arr = np.zeros(k_max, dtype=np.float32)
        for i, arm in enumerate(self.config.arms):
            eu = calc_est_units(arm, req.input_tokens, req.expected_output_tokens)
            est_units_arr[i] = np.float32(eu)

        util_matrix = self.ledger.utilization(now)

        qv = make_quota_view(util_matrix, est_units_arr, allowed_mask)

        x_jax = to_jax_array(x_np)
        state_next, arm_idx, propensity, diag = select_jit(self.state, x_jax, qv, self.static)

        # Abort if non-finite scores outside JIT
        n_arms = len(self.config.arms)
        active_scores = np.array(diag.score[:n_arms])
        policy_active = np.array(diag.policy[:n_arms]) > 0
        if not np.all(np.isfinite(np.where(policy_active, active_scores, 0.0))):
            _log_event(
                sys.stderr, "error", "non_finite_score", error="Non-finite score detected in select"
            )
            raise NumericalIntegrityError("Non-finite score detected in select")

        if bool(diag.no_arm_available):
            masked_reason: dict[str, str] = {}
            for i, arm in enumerate(self.config.arms):
                if not allowed_mask[i]:
                    masked_reason[arm.arm_id] = "not_allowed"
                elif np.any(util_matrix[i] >= 1.0):
                    masked_reason[arm.arm_id] = "quota_exhausted"
                else:
                    masked_reason[arm.arm_id] = "disabled"
            raise RpcError(-32004, "No arm available", data={"masked_reason": masked_reason})

        arm_i = int(arm_idx)
        chosen_arm = self.config.arms[arm_i]
        prop_val = float(propensity)
        d_id = uuid7()
        est_u = float(est_units_arr[arm_i])

        eid = self.ledger.peek_next_event_id()

        # Write WAL line select BEFORE mutating state or ledger (FR-20 / MF-2)
        util_dict = {
            arm.arm_id: util_matrix[idx, : len(arm.windows)].tolist()
            for idx, arm in enumerate(self.config.arms)
        }
        score_logged = np.where(np.array(diag.policy) > 0, np.array(diag.score), -1e9)
        select_entry = {
            "phase": "select",
            "decision_id": d_id,
            "ts": now,
            "state_version": self.state_version,
            "arm_index": arm_i,
            "arm_id": chosen_arm.arm_id,
            "propensity": prop_val,
            "alpha": self.static.alpha,
            "x": x_np.tolist(),
            "policy": np.array(diag.policy).tolist(),
            "score": score_logged.tolist(),
            "exploit": np.array(diag.exploit).tolist(),
            "bonus": np.array(diag.bonus).tolist(),
            "shadow_price": np.array(diag.shadow_price).tolist(),
            "utilization": util_dict,
            "est_units": est_u,
            "ledger_event_id": eid,
            "featurizer_version": FEATURIZER_VERSION,
        }
        if self.dlog is not None:
            self.dlog.append(select_entry)

        # Mutate ledger and state only AFTER WAL append succeeds
        rec_eid = self.ledger.record(chosen_arm.arm_id, est_u, now)
        assert rec_eid == eid
        self.state = state_next

        pending = Pending(
            decision_id=d_id,
            arm_index=arm_i,
            x=x_np.tolist(),
            propensity=prop_val,
            ts_select=now,
            est_units=est_u,
            ledger_event_id=eid,
            state_version=self.state_version,
            raw_entry=select_entry,
        )
        self.pending_store.put(pending)
        self._maybe_snapshot()

        res = {
            "decision_id": d_id,
            "arm": {
                "arm_id": chosen_arm.arm_id,
                "model_id": chosen_arm.model_id,
                "provider": chosen_arm.provider,
                "billing_mode": chosen_arm.billing_mode,
            },
            "propensity": prop_val,
        }

        if params.get("verbose", True):
            res["diagnostics"] = _build_select_diagnostics(
                self.config.arms, diag, util_matrix, est_units_arr
            )

        return res

    def report(self, params: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates report step (FR-21, FR-28)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        d_id = params.get("decision_id")
        if not isinstance(d_id, str):
            raise RpcError(-32602, "Missing decision_id", data={"field": "decision_id"})

        quality_val = _check_finite_float(
            params.get("quality"), "quality", min_val=0.0, max_val=1.0
        )
        cost_val = _check_finite_float(params.get("cost_usd", 0.0), "cost_usd", min_val=0.0)
        lat_val = _check_finite_float(params.get("latency_s", 0.0), "latency_s", min_val=0.0)

        in_toks = _check_non_neg_int(params.get("actual_input_tokens", 0), "actual_input_tokens")
        out_toks = _check_non_neg_int(params.get("actual_output_tokens", 0), "actual_output_tokens")

        now = self.clock()
        self._check_clock_skew(now)

        pending = self.pending_store.get(d_id)
        if pending is not None:
            if (now - pending.ts_select) > self.config.bandit.pending_ttl_s:
                self.pending_store.pop(d_id)
                expired_at = pending.ts_select + float(self.config.bandit.pending_ttl_s)
                self.pending_store.mark_expired(d_id, expired_at)
                if self.dlog is not None:
                    self.dlog.append(
                        {
                            "phase": "expire",
                            "decision_id": d_id,
                            "ts": now,
                            "state_version": self.state_version,
                            "ts_select": pending.ts_select,
                        }
                    )
                raise RpcError(
                    -32003,
                    f"Decision expired: {d_id}",
                    data={"expired_at": expired_at},
                )
        else:
            if self.pending_store.is_expired(d_id):
                expired_at = self.pending_store.get_expired_at(d_id) or 0.0
                raise RpcError(
                    -32003,
                    f"Decision expired: {d_id}",
                    data={"expired_at": expired_at},
                )
            if self.pending_store.is_applied(d_id):
                _log_event(
                    sys.stderr,
                    "info",
                    "report_duplicate",
                    decision_id=d_id,
                    state_version=self.state_version,
                )
                return {
                    "applied": False,
                    "reason": "duplicate",
                    "state_version": self.state_version,
                }
            raise RpcError(-32002, f"Unknown decision_id: {d_id}")

        arm = self.config.arms[pending.arm_index]

        # Subscription arm cost restriction (FR-19)
        if arm.billing_mode == "subscription" and cost_val > 0.0:
            msg = "cost_usd must be 0 for subscription arms"
            raise RpcError(-32602, msg, data={"field": "cost_usd"})

        actual_units = calc_est_units(arm, in_toks, out_toks)
        reward = compute_reward(quality_val, cost_val, lat_val, self.static)

        next_state_version = self.state_version + 1

        report_entry = {
            "phase": "report",
            "decision_id": d_id,
            "ts": now,
            "state_version": next_state_version,
            "quality": quality_val,
            "cost_usd": cost_val,
            "latency_s": lat_val,
            "actual_input_tokens": in_toks,
            "actual_output_tokens": out_toks,
            "reward": reward,
            "actual_units": actual_units,
        }
        # FR-21 exact order: append WAL line FIRST (with next state_version)
        if self.dlog is not None:
            self.dlog.append(report_entry)

        self.ledger.correct(pending.ledger_event_id, arm.arm_id, actual_units)
        self.state_version = next_state_version
        x_jax = to_jax_array(pending.x)
        self.state = update_jit(self.state, x_jax, pending.arm_index, reward, self.static)

        self.pending_store.pop(d_id)
        self.pending_store.mark_applied(d_id)
        self.updates_since_snapshot += 1

        self._maybe_snapshot()

        return {
            "applied": True,
            "reward": reward,
            "state_version": self.state_version,
        }

    def _maybe_snapshot(self) -> None:
        now = self.clock()
        should_snapshot = (
            self.updates_since_snapshot >= self.config.bandit.snapshot_every_updates
            or (now - self.last_snapshot_ts) >= self.config.bandit.snapshot_every_s
        )
        needs_rot = self.dlog is not None and self.dlog.needs_rotation

        if should_snapshot or needs_rot:
            self.snapshot({})

        if needs_rot and self.dlog is not None:
            carry_forward = [
                p.raw_entry for p in self.pending_store.values() if p.raw_entry is not None
            ]
            bytes_before = self.dlog.path.stat().st_size if self.dlog.path.exists() else 0
            self.dlog.rotate(carry_forward=carry_forward)
            _log_event(
                sys.stderr,
                "info",
                "wal_rotated",
                bytes=bytes_before,
                carried_forward=len(carry_forward),
            )

    def snapshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Barries expired and writes state snapshot (FR-25, FR-28)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        now = self.clock()
        expired = self.pending_store.sweep(now, self.config.bandit.pending_ttl_s)
        for p in expired:
            if self.dlog is not None:
                self.dlog.append(
                    {
                        "phase": "expire",
                        "decision_id": p.decision_id,
                        "ts": now,
                        "state_version": self.state_version,
                        "ts_select": p.ts_select,
                    }
                )

        meta = SnapshotMeta(
            schema_version=1,
            state_version=self.state_version,
            featurizer_version=FEATURIZER_VERSION,
            arm_ids=[arm.arm_id for arm in self.config.arms],
            static=self.static,
            seed=self.config.bandit.seed,
            created_at=datetime.datetime.fromtimestamp(now, tz=datetime.UTC).isoformat(),
            ledger=self.ledger.dump(),
            config_sha256=compute_config_sha256(self.config),
        )

        target_path, n_bytes = write_snapshot(self.snapshot_dir, self.state, self.ledger, meta)
        self.updates_since_snapshot = 0
        self.last_snapshot_ts = now

        _log_event(
            sys.stderr,
            "info",
            "snapshot_written",
            timestamp=now,
            state_version=self.state_version,
            path=str(target_path),
            bytes=n_bytes,
        )

        return {
            "state_version": self.state_version,
            "path": str(target_path),
            "bytes": n_bytes,
            "expired_pending": len(expired),
        }

    def health(self, params: dict[str, Any]) -> dict[str, Any]:
        """Returns health metrics and JIT compilation count (FR-31)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        now = self.clock()
        util_matrix = self.ledger.utilization(now)

        arm_health: dict[str, Any] = {}
        for i, arm in enumerate(self.config.arms):
            arm_health[arm.arm_id] = {
                "mask": bool(self.state.mask[i]),
                "n_pulls": int(self.state.n_pulls[i]),
                "utilization": util_matrix[i, : len(arm.windows)].tolist(),
            }

        return {
            "ready": True,
            "state_version": self.state_version,
            "pending": len(self.pending_store),
            "updates_since_snapshot": self.updates_since_snapshot,
            "arms": arm_health,
            "jit_compilations": dict(_compile_count),
        }

    def close(self) -> int:
        exit_code = 0
        if self.ready:
            try:
                self.snapshot({})
            except Exception as err:
                _log_event(
                    sys.stderr,
                    "error",
                    "shutdown_snapshot_failed",
                    error=str(err),
                    state_version=self.state_version,
                )
                exit_code = 4
        if self.dlog is not None:
            try:
                self.dlog.close()
            except Exception as err:
                _log_event(
                    sys.stderr,
                    "error",
                    "shutdown_dlog_close_failed",
                    error=str(err),
                )
                exit_code = 4
        return exit_code


def _read_line_bounded(stream: Any, limit: int = 1048576) -> tuple[str | None, bool]:
    """Reads line up to limit bytes from stream. Drains remainder if oversized."""
    buf = getattr(stream, "buffer", stream)
    if hasattr(buf, "readline"):
        raw = buf.readline(limit + 1)
        if not raw:
            return None, False
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
        if len(raw_bytes) > limit:
            while True:
                chunk = buf.readline(4096)
                if not chunk:
                    break
                chunk_bytes = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                if b"\n" in chunk_bytes:
                    break
            return None, True
        return raw_bytes.decode("utf-8", errors="replace"), False
    else:
        line = stream.readline()
        if not line:
            return None, False
        line_bytes = line.encode("utf-8")
        if len(line_bytes) > limit:
            return None, True
        return line, False


def _dispatch_rpc_method(router: Router, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "select":
        return router.select(params)
    if method == "report":
        return router.report(params)
    if method == "snapshot":
        return router.snapshot(params)
    if method == "health":
        return router.health(params)
    raise RpcError(-32601, f"Method not found: {method}")


def serve(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    router: Router,
    exit_on_eof: bool = True,
) -> None:
    """Reads JSON-RPC 2.0 requests line by line from stdin and writes to stdout (FR-29)."""
    while True:
        line_str, is_too_long = _read_line_bounded(stdin, limit=1048576)
        if line_str is None and not is_too_long:
            code = router.close()
            if exit_on_eof:
                sys.exit(code)
            break

        if is_too_long:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid request",
                    "data": {"reason": "line_too_long"},
                },
            }
            stdout.write(json.dumps(err_resp) + "\n")
            stdout.flush()
            continue

        if not line_str or not line_str.strip():
            continue

        try:
            payload = json.loads(line_str.strip())
        except Exception:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            stdout.write(json.dumps(err_resp) + "\n")
            stdout.flush()
            continue

        if isinstance(payload, list):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Batch requests are not supported"},
            }
            stdout.write(json.dumps(err_resp) + "\n")
            stdout.flush()
            continue

        is_dict_payload = isinstance(payload, dict)
        is_valid_rpc = is_dict_payload and payload.get("jsonrpc") == "2.0" and "method" in payload
        if not is_valid_rpc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": payload.get("id") if isinstance(payload, dict) else None,
                "error": {"code": -32600, "message": "Invalid request"},
            }
            stdout.write(json.dumps(err_resp) + "\n")
            stdout.flush()
            continue

        req_id = payload.get("id")
        method = payload["method"]
        params = payload.get("params", {})

        try:
            result = _dispatch_rpc_method(router, method, params)

            if req_id is not None:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                try:
                    out = json.dumps(resp, allow_nan=False) + "\n"
                except ValueError as err:
                    msg = f"Non-finite float in response serialization: {err}"
                    raise NumericalIntegrityError(msg) from err
                stdout.write(out)
                stdout.flush()

        except (JournalWriteError, NumericalIntegrityError) as err:
            _log_event(stderr, "error", type(err).__name__.lower(), error=str(err))
            code = router.close()
            sys.exit(code or 4)
        except RpcError as err:
            err_obj: dict[str, Any] = {"code": err.code, "message": err.message}
            if err.data is not None:
                err_obj["data"] = err.data

            if req_id is not None:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": err_obj}
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()
            else:
                _log_event(stderr, "error", "rpc_notification_error", error=err_obj)
        except Exception as err:
            _log_event(stderr, "error", "unhandled_error", error=str(err))
            if req_id is not None:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Internal error: {err}"},
                }
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="azuma-router JSON-RPC 2.0 daemon")
    parser.add_argument("--config", required=True, help="Path to azuma.toml")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as err:
        _log_event(sys.stderr, "error", "config_error", error=str(err))
        sys.exit(3)

    router = Router(cfg)

    def handle_signal(_sig: int, _frame: Any) -> None:
        _log_event(sys.stderr, "info", "signal_received", signal=_sig)
        code = router.close()
        sys.exit(code)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    ready_params = router.boot()
    ready_notif = {"jsonrpc": "2.0", "method": "ready", "params": ready_params}
    sys.stdout.write(json.dumps(ready_notif) + "\n")
    sys.stdout.flush()

    serve(sys.stdin, sys.stdout, sys.stderr, router)


if __name__ == "__main__":
    main()
