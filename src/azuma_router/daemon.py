"""Single-thread JSON-RPC 2.0 daemon over stdio.

Complies with contract §3.8, §5, FR-29..FR-33, ADR-0010, ADR-0013, ADR-0014.
"""

import argparse
import datetime
import json
import secrets
import signal
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import jax.numpy as jnp
import numpy as np

from azuma_router.config import Config, load_config, to_static_config
from azuma_router.errors import (
    ConfigError,
    InvalidContextError,
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
from azuma_router.state import init_state
from azuma_router.types import (
    F32,
    F64,
    BanditState,
    Pending,
    QuotaView,
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
                                    sys.stderr.write(f"[ERROR] {msg}\n")
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
            )
        except ValueError as err:
            sys.stderr.write(f"[ERROR] Replay failed: {err}\n")
            sys.exit(3)

        self.pending_store = p_store

        # Re-apply reports subsequent to snapshot
        for s_ver, x_arr, arm_idx, reward in applied_reports:
            self.state = update_jit(
                self.state,
                jnp.array(x_arr, dtype=F64),
                arm_idx,
                reward,
                self.static,
            )
            self.state_version = max(self.state_version, s_ver)

        # 3. Open DecisionLog
        self.dlog = DecisionLog(self.decision_log_path, self.config.bandit.decision_log_max_bytes)

        # 4. Warm-up JIT
        t_start = time.perf_counter()
        dummy_x = jnp.zeros(32, dtype=F64)
        dummy_qv = QuotaView(
            utilization=jnp.zeros((self.static.k_max, self.static.w_max), dtype=F32),
            est_units=jnp.zeros(self.static.k_max, dtype=F32),
            allowed=jnp.ones(self.static.k_max, dtype=bool),
        )

        self.state, _, _, _ = select_jit(self.state, dummy_x, dummy_qv, self.static)
        self.state = update_jit(self.state, dummy_x, 0, 0.0, self.static)
        warmup_ms = (time.perf_counter() - t_start) * 1000.0

        self.ready = True

        return {
            "state_version": self.state_version,
            "arms": [arm.arm_id for arm in self.config.arms],
            "restored_from_snapshot": restored_from_snapshot,
            "pending_replayed": len(self.pending_store),
            "warmup_ms": warmup_ms,
        }

    def _check_clock_skew(self, now: float) -> None:
        if now < self.last_ts - 1.0:
            skew_log = {
                "ts": now,
                "level": "warning",
                "event": "clock_skew",
                "delta_s": now - self.last_ts,
            }
            sys.stderr.write(json.dumps(skew_log) + "\n")
            sys.stderr.flush()
        self.last_ts = now

    def _sweep_expired(self) -> None:
        """Sweeps expired pending decisions and writes expire WAL lines."""
        now = self.clock()
        self._check_clock_skew(now)
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
                has_code=bool(req_raw.get("has_code", False)),
                has_tools=bool(req_raw.get("has_tools", False)),
                has_images=bool(req_raw.get("has_images", False)),
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
        allowed_list = params.get("allowed_arms")
        allowed_mask = [False] * k_max

        all_arm_ids = [arm.arm_id for arm in self.config.arms]
        if allowed_list is not None:
            if not isinstance(allowed_list, list):
                raise RpcError(
                    -32602, "allowed_arms must be a list", data={"field": "allowed_arms"}
                )
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

        # est_units
        est_units_arr = np.zeros(k_max, dtype=np.float32)
        for i, arm in enumerate(self.config.arms):
            eu = calc_est_units(arm, req.input_tokens, req.expected_output_tokens)
            est_units_arr[i] = np.float32(eu)

        util_matrix = self.ledger.utilization(now)

        qv = QuotaView(
            utilization=jnp.array(util_matrix, dtype=F32),
            est_units=jnp.array(est_units_arr, dtype=F32),
            allowed=jnp.array(allowed_mask, dtype=bool),
        )

        x_jax = jnp.array(x_np, dtype=F64)
        self.state, arm_idx, propensity, diag = select_jit(self.state, x_jax, qv, self.static)

        # Abort if non-finite scores outside JIT
        n_arms = len(self.config.arms)
        active_scores = np.array(diag.score[:n_arms])
        policy_active = np.array(diag.policy[:n_arms]) > 0
        if not np.all(np.isfinite(np.where(policy_active, active_scores, 0.0))):
            sys.stderr.write("[ERROR] Non-finite score detected in select\n")
            sys.exit(4)

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

        eid = self.ledger.record(chosen_arm.arm_id, est_u, now)

        # Write WAL line select BEFORE returning (FR-20)
        util_dict = {
            arm.arm_id: util_matrix[idx, : len(arm.windows)].tolist()
            for idx, arm in enumerate(self.config.arms)
        }
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
            "score": np.array(diag.score).tolist(),
            "exploit": np.array(diag.exploit).tolist(),
            "bonus": np.array(diag.bonus).tolist(),
            "shadow_price": np.array(diag.shadow_price).tolist(),
            "utilization": util_dict,
            "est_units": est_u,
            "featurizer_version": FEATURIZER_VERSION,
        }
        if self.dlog is not None:
            self.dlog.append(select_entry)

        pending = Pending(
            decision_id=d_id,
            arm_index=arm_i,
            x=x_np.tolist(),
            propensity=prop_val,
            ts_select=now,
            est_units=est_u,
            ledger_event_id=eid,
            state_version=self.state_version,
        )
        self.pending_store.put(pending)

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
            arms = self.config.arms
            diag_dict = {
                "score": {arm.arm_id: float(diag.score[i]) for i, arm in enumerate(arms)},
                "exploit": {arm.arm_id: float(diag.exploit[i]) for i, arm in enumerate(arms)},
                "bonus": {arm.arm_id: float(diag.bonus[i]) for i, arm in enumerate(arms)},
                "shadow_price": {
                    arm.arm_id: float(diag.shadow_price[i]) for i, arm in enumerate(arms)
                },
                "policy": {arm.arm_id: float(diag.policy[i]) for i, arm in enumerate(arms)},
                "utilization": {
                    arm.arm_id: {
                        f"{w.kind}:{w.duration_s or w.period}": float(util_matrix[i, j])
                        for j, w in enumerate(arm.windows)
                    }
                    for i, arm in enumerate(arms)
                },
                "est_units": {
                    arm.arm_id: float(est_units_arr[i]) for i, arm in enumerate(arms)
                },
            }
            res["diagnostics"] = diag_dict

        return res

    def report(self, params: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates report step (FR-21, FR-28)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        d_id = params.get("decision_id")
        if not isinstance(d_id, str):
            raise RpcError(-32602, "Missing decision_id", data={"field": "decision_id"})

        quality = params.get("quality")
        if quality is None or not isinstance(quality, float | int) or isinstance(quality, bool):
            msg = "quality must be float in [0, 1]"
            raise RpcError(-32602, msg, data={"field": "quality"})
        quality_val = float(quality)
        if not (0.0 <= quality_val <= 1.0):
            msg = "quality must be float in [0, 1]"
            raise RpcError(-32602, msg, data={"field": "quality"})

        cost_usd = params.get("cost_usd", 0.0)
        if not isinstance(cost_usd, float | int) or float(cost_usd) < 0.0:
            raise RpcError(-32602, "cost_usd must be >= 0", data={"field": "cost_usd"})
        cost_val = float(cost_usd)

        latency_s = params.get("latency_s", 0.0)
        if not isinstance(latency_s, float | int) or float(latency_s) < 0.0:
            raise RpcError(-32602, "latency_s must be >= 0", data={"field": "latency_s"})
        lat_val = float(latency_s)

        in_toks = params.get("actual_input_tokens", 0)
        if not isinstance(in_toks, int) or in_toks < 0:
            msg = "actual_input_tokens must be int >= 0"
            raise RpcError(-32602, msg, data={"field": "actual_input_tokens"})

        out_toks = params.get("actual_output_tokens", 0)
        if not isinstance(out_toks, int) or out_toks < 0:
            msg = "actual_output_tokens must be int >= 0"
            raise RpcError(-32602, msg, data={"field": "actual_output_tokens"})

        pending = self.pending_store.get(d_id)
        if pending is None:
            if self.pending_store.is_applied(d_id):
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

        now = self.clock()
        self._check_clock_skew(now)
        actual_units = calc_est_units(arm, in_toks, out_toks)
        self.ledger.correct(pending.ledger_event_id, arm.arm_id, actual_units)

        reward = compute_reward(quality_val, cost_val, lat_val, self.static)

        # State version is incremented BEFORE writing report WAL line (FR-21)
        self.state_version += 1

        report_entry = {
            "phase": "report",
            "decision_id": d_id,
            "ts": now,
            "state_version": self.state_version,
            "quality": quality_val,
            "cost_usd": cost_val,
            "latency_s": lat_val,
            "actual_input_tokens": in_toks,
            "actual_output_tokens": out_toks,
            "reward": reward,
            "actual_units": actual_units,
        }
        if self.dlog is not None:
            self.dlog.append(report_entry)

        x_jax = jnp.array(pending.x, dtype=F64)
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
        if (
            self.updates_since_snapshot >= self.config.bandit.snapshot_every_updates
            or (now - self.last_snapshot_ts) >= self.config.bandit.snapshot_every_s
        ):
            self.snapshot({})

    def snapshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Barries expired and writes state snapshot (FR-25, FR-28)."""
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

    def close(self) -> None:
        if self.ready:
            try:
                self.snapshot({})
            except Exception:
                pass
        if self.dlog is not None:
            self.dlog.close()


def serve(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    router: Router,
    exit_on_eof: bool = True,
) -> None:
    """Reads JSON-RPC 2.0 requests line by line from stdin and writes to stdout (FR-29)."""
    while True:
        line = stdin.readline()
        if not line:
            if exit_on_eof:
                router.close()
                sys.exit(0)
            break

        line_str = line.strip()
        if not line_str:
            continue

        if len(line) > 1048576:
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

        try:
            payload = json.loads(line_str)
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
            if method == "select":
                result = router.select(params)
            elif method == "report":
                result = router.report(params)
            elif method == "snapshot":
                result = router.snapshot(params)
            elif method == "health":
                result = router.health(params)
            else:
                raise RpcError(-32601, f"Method not found: {method}")

            if req_id is not None:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()

        except RpcError as err:
            err_obj: dict[str, Any] = {"code": err.code, "message": err.message}
            if err.data is not None:
                err_obj["data"] = err.data

            if req_id is not None:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": err_obj}
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()
            else:
                stderr.write(f"[RPC Notification Error] {err_obj}\n")
                stderr.flush()
        except Exception as err:
            stderr.write(f"[Unhandled Error] {err}\n")
            stderr.flush()
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
        sys.stderr.write(f"[ERROR] Config loading failed: {err}\n")
        sys.exit(3)

    router = Router(cfg)

    def handle_signal(_sig: int, _frame: Any) -> None:
        sys.stderr.write("[INFO] Received signal, shutting down cleanly...\n")
        router.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    ready_params = router.boot()
    ready_notif = {"jsonrpc": "2.0", "method": "ready", "params": ready_params}
    sys.stdout.write(json.dumps(ready_notif) + "\n")
    sys.stdout.flush()

    serve(sys.stdin, sys.stdout, sys.stderr, router)


if __name__ == "__main__":
    main()
