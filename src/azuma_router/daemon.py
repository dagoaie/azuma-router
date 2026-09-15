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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn, TextIO

import numpy as np

from azuma_router.config import Config, load_config, to_static_config
from azuma_router.errors import (
    AzumaError,
    ConfigError,
    IncompatibleSnapshotError,
    InvalidContextError,
    InvalidRequestLineError,
    JournalApplyError,
    JournalWriteError,
    LedgerEventNotFoundError,
    NumericalIntegrityError,
    RpcError,
    SnapshotWriteError,
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
    U32,
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


_MASKED_DIAGNOSTICS = ("score", "exploit", "bonus", "shadow_price")


def _build_select_diagnostics(
    arms: tuple[Any, ...],
    diag: Any,
    util_matrix: np.ndarray,
    est_units_arr: np.ndarray,
) -> dict[str, Any]:
    policies = [float(diag.policy[i]) for i in range(len(arms))]

    def _masked_by_policy(values: Any) -> dict[str, float | None]:
        """Per-arm values; arms the policy masked out (policy == 0) report null (§4.5)."""
        return {
            arm.arm_id: float(values[i]) if policies[i] > 0 else None for i, arm in enumerate(arms)
        }

    return {
        **{name: _masked_by_policy(getattr(diag, name)) for name in _MASKED_DIAGNOSTICS},
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


def _exit_unrecoverable(router: "Router", event: str, reason: object) -> NoReturn:
    """Boot cannot continue without operator action: structured stderr event, exit 3 (ADR-0014).

    `closing` is latched first so a signal landing before `sys.exit` cannot turn the exit
    code into 0 through the coordinator (P6-RR-2)."""
    router.closing = True
    _log_event(sys.stderr, "error", event, error=str(reason))
    sys.exit(3)


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
        self.closing = False
        self.state_version = 0
        self.selection_sequence = 0
        self.updates_since_snapshot = 0
        self.last_snapshot_ts = self.clock()
        self.last_ts = self.clock()

        self.state: BanditState
        self.ledger: QuotaLedger
        self.pending_store = PendingStore()
        self.dlog: DecisionLog | None = None

    def boot(self) -> dict[str, Any]:
        """Initializes or restores state, replays WAL, warms up JIT, and marks ready (FR-30)."""
        r_name = f"{self.decision_log_path.stem}.1{self.decision_log_path.suffix}"
        rotated_path = self.decision_log_path.with_name(r_name)

        # 1. Restore snapshot or clean init
        try:
            res_snap = restore_snapshot(self.snapshot_dir, self.config)
        except IncompatibleSnapshotError as err:
            _exit_unrecoverable(self, "incompatible_snapshot", err)
        except OSError as err:
            _exit_unrecoverable(self, "snapshot_read_failed", err)
        restored_state, restored_ledger, restored_meta = res_snap

        if restored_state is not None and restored_ledger is not None and restored_meta is not None:
            self.state = restored_state
            self.ledger = restored_ledger
            self.state_version = restored_meta.state_version
            self.selection_sequence = restored_meta.selection_sequence
            restored_from_snapshot = True
        else:
            self._refuse_learned_journal_without_snapshot(rotated_path)
            self.state = init_state(self.config)
            self.ledger = QuotaLedger(self.config)
            self.state_version = 0
            self.selection_sequence = 0
            restored_from_snapshot = False

        # 2. Replay WAL
        try:
            (
                p_store,
                applied_reports,
                _applied_ids,
                replayed_rng_key,
                replayed_selection_sequence,
            ) = replay_log(
                active_path=self.decision_log_path,
                rotated_path=rotated_path if rotated_path.exists() else None,
                snapshot_state_version=self.state_version,
                snapshot_selection_sequence=self.selection_sequence,
                config=self.config,
                ledger=self.ledger,
            )
        except (ValueError, OSError) as err:
            _exit_unrecoverable(self, "replay_failed", err)

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
        if replayed_rng_key is not None:
            self.state = self.state._replace(rng_key=to_jax_array(replayed_rng_key, dtype=U32))
        self.selection_sequence = replayed_selection_sequence

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

    def _refuse_learned_journal_without_snapshot(self, rotated_path: Path) -> None:
        """Rotation always follows a snapshot, so a rotated segment without snapshot/ means the
        snapshot was lost (P3-MF-2). An unrotated journal replays whole from the initial state,
        which is exactly the case of a kill before the first cadence snapshot (P5-MF-1)."""
        if rotated_path.exists():
            msg = "journal was rotated (which always follows a snapshot) but no snapshot/ exists"
            _exit_unrecoverable(self, "wal_inconsistent", msg)

    def _journal_expire(self, decision_id: str, ts_select: float, now: float) -> None:
        """Appends the single expire WAL line shape shared by every expiry path (P3-SF-1)."""
        if self.dlog is None:
            return
        self.dlog.append(
            {
                "phase": "expire",
                "decision_id": decision_id,
                "ts": now,
                "state_version": self.state_version,
                "ts_select": ts_select,
            }
        )

    def _sweep_expired(self, now: float) -> list[Pending]:
        """Sweeps expired pending decisions and journals one expire line each."""
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
            self._journal_expire(p.decision_id, p.ts_select, now)
        return expired

    def select(self, params: dict[str, Any]) -> dict[str, Any]:
        """Orchestrates select step (FR-20, FR-31)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        now = self.clock()
        self._sweep_expired(now)
        # Expire appends count towards rotation even if this request is rejected (R4-SF-2).
        self._rotate_if_needed()

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

        # Abort if non-finite scores outside JIT. Gate by the structural `active` mask, not by
        # `policy > 0` (NUM-1): an active arm's policy can legitimately underflow to exactly zero
        # under epsilon_floor = 0, which would otherwise mask a NaN score as harmless.
        n_arms = len(self.config.arms)
        active_arr = np.array(diag.active)
        active_scores = np.array(diag.score[:n_arms])
        if not np.all(np.isfinite(np.where(active_arr[:n_arms], active_scores, 0.0))):
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
        next_selection_sequence = self.selection_sequence + 1
        d_id = uuid7()
        est_u = float(est_units_arr[arm_i])

        eid = self.ledger.next_event_id()

        # Write WAL line select BEFORE mutating state or ledger (FR-20 / MF-2)
        util_dict = {
            arm.arm_id: util_matrix[idx, : len(arm.windows)].tolist()
            for idx, arm in enumerate(self.config.arms)
        }
        # Sentinel by the real active mask, not by a possibly-underflowed policy (OPE-2): an
        # active arm's score must survive even when epsilon_floor gave it zero policy mass.
        score_logged = np.where(active_arr, np.array(diag.score), -1e9)
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
            "radius": np.array(diag.radius).tolist(),
            "active": active_arr.tolist(),
            "shadow_price": np.array(diag.shadow_price).tolist(),
            "utilization": util_dict,
            "est_units": est_u,
            "ledger_event_id": eid,
            "featurizer_version": FEATURIZER_VERSION,
            "selection_sequence": next_selection_sequence,
            "rng_key": np.asarray(state_next.rng_key, dtype=np.uint32).tolist(),
        }
        if self.dlog is not None:
            self.dlog.append(select_entry)

        # Mutate ledger and state only AFTER WAL append succeeds
        self.ledger.record(eid, chosen_arm.arm_id, est_u, now)
        self.state = state_next
        self.selection_sequence = next_selection_sequence

        pending = Pending(
            decision_id=d_id,
            arm_index=arm_i,
            x=x_np.tolist(),
            propensity=prop_val,
            ts_select=now,
            est_units=est_u,
            ledger_event_id=eid,
            state_version=self.state_version,
            selection_sequence=next_selection_sequence,
            arm_id=chosen_arm.arm_id,
            select_entry=select_entry,
        )
        self.pending_store.put(pending)
        self._rotate_if_needed()

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
                self._journal_expire(d_id, pending.ts_select, now)
                self._rotate_if_needed()
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
        # A live pending must still own its ledger event; otherwise state is corrupt (P3-MF-1)
        if not self.ledger.has_event(pending.ledger_event_id):
            raise LedgerEventNotFoundError(pending.ledger_event_id, arm.arm_id)

        # FR-21 exact order: append WAL line FIRST (with next state_version)
        if self.dlog is not None:
            self.dlog.append(report_entry)

        # The flushed line survives a process crash. From here on, memory must reach the journaled
        # state or the process must die so replay reconciles it on restart (P5-MF-1).
        try:
            self.ledger.correct(pending.ledger_event_id, arm.arm_id, actual_units)
            self.state_version = next_state_version
            x_jax = to_jax_array(pending.x)
            self.state = update_jit(self.state, x_jax, pending.arm_index, reward, self.static)
            self.pending_store.pop(d_id)
            self.pending_store.mark_applied(d_id)
        except Exception as err:
            msg = (
                f"report {d_id} journaled as state_version {next_state_version} not applied: {err}"
            )
            raise JournalApplyError(msg) from err
        self.updates_since_snapshot += 1

        self._maybe_snapshot()
        self._rotate_if_needed()

        return {
            "applied": True,
            "reward": reward,
            "state_version": self.state_version,
        }

    def _maybe_snapshot(self) -> None:
        """Takes the FR-28 cadence snapshot (by update count or elapsed time) after an update."""
        now = self.clock()
        due_by_updates = self.updates_since_snapshot >= self.config.bandit.snapshot_every_updates
        due_by_time = (now - self.last_snapshot_ts) >= self.config.bandit.snapshot_every_s
        if due_by_updates or due_by_time:
            self.snapshot({})

    def _rotate_if_needed(self) -> None:
        """Snapshots once the WAL exceeds max bytes; snapshot() then rotates the journal."""
        if self.dlog is not None and self.dlog.needs_rotation():
            self.snapshot({})

    def _rotate_after_snapshot(self) -> None:
        """Rotates the WAL right after a snapshot, so no journaled update is ever dropped."""
        if self.dlog is None or not self.dlog.needs_rotation():
            return
        carry_forward = [p.select_entry for p in self.pending_store.values()]
        bytes_before = self.dlog.path.stat().st_size
        self.dlog.rotate(carry_forward=carry_forward)
        _log_event(
            sys.stderr,
            "info",
            "wal_rotated",
            bytes=bytes_before,
            carried_forward=len(carry_forward),
        )

    def snapshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Journals expiries, writes the snapshot, then rotates the WAL if due (FR-25, FR-28)."""
        if not self.ready:
            raise RpcError(-32005, "Daemon not ready")

        now = self.clock()
        expired = self._sweep_expired(now)

        meta = SnapshotMeta(
            schema_version=2,
            state_version=self.state_version,
            featurizer_version=FEATURIZER_VERSION,
            arm_ids=[arm.arm_id for arm in self.config.arms],
            static=self.static,
            seed=self.config.bandit.seed,
            selection_sequence=self.selection_sequence,
            created_at=datetime.datetime.fromtimestamp(now, tz=datetime.UTC).isoformat(),
            ledger=self.ledger.dump(),
            config_sha256=compute_config_sha256(self.config),
        )

        if self.dlog is not None:
            # A pre-publication sync failure is a snapshot-step failure (FR-25): translate it here
            # so it is never misfiled as journal_write_error and retried through close() (SNAP-1).
            try:
                self.dlog.sync()
            except JournalWriteError as err:
                raise SnapshotWriteError(f"snapshot sync failed: {err}") from err
        target_path, n_bytes = write_snapshot(self.snapshot_dir, self.state, self.ledger, meta)
        self.updates_since_snapshot = 0
        self.last_snapshot_ts = now
        self._rotate_after_snapshot()

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

    def close(self, *, with_snapshot: bool = True) -> int:
        """Flushes and closes; the shutdown snapshot is skipped when state may be corrupt.

        `closing` is set first so a signal landing mid-close is ignored instead of starting a
        second close with a snapshot the fatal path forbids (P5-MF-2)."""
        self.closing = True
        exit_code = 0
        if self.ready and with_snapshot:
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


# Fatal error -> (stderr event, whether a shutdown snapshot is still safe). A journal write
# failure leaves the in-memory state consistent (WAL-first, nothing applied); the others do not
# or cannot persist it (contract: numerical corruption is never snapshotted).
_FATAL_POLICY: dict[type[AzumaError], tuple[str, bool]] = {
    JournalWriteError: ("journal_write_error", True),
    JournalApplyError: ("journal_apply_error", False),
    SnapshotWriteError: ("snapshot_write_error", False),
    NumericalIntegrityError: ("numerical_integrity_error", False),
    LedgerEventNotFoundError: ("ledger_integrity_error", False),
}
_FATAL_ERRORS = tuple(_FATAL_POLICY)


def _exit_fatal(stderr: TextIO, router: Router, err: AzumaError, phase: str) -> NoReturn:
    """Logs a fatal integrity error, closes the router and exits 4 (ADR-0014)."""
    event_name, with_snapshot = _FATAL_POLICY[type(err)]
    router.closing = True  # a signal from here on must not replace exit 4 (P6-RR-2)
    _log_event(stderr, "error", event_name, phase=phase, error=str(err))
    router.close(with_snapshot=with_snapshot)
    sys.exit(4)


def _error_response(
    req_id: str | int | None, code: int, message: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _write_response(stdout: TextIO, response: dict[str, Any]) -> None:
    """Every stdout line is strict JSON; a non-finite float anywhere is an integrity failure."""
    try:
        out = json.dumps(response, allow_nan=False) + "\n"
    except ValueError as err:
        msg = f"Non-finite float in response serialization: {err}"
        raise NumericalIntegrityError(msg) from err
    stdout.write(out)
    stdout.flush()


def _reject_json_constant(name: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {name}")


def _validate_request(line_str: str) -> dict[str, Any]:
    """Parses one line as a JSON-RPC 2.0 request; rejects NaN/Infinity and bad ids (P3-MF-4)."""
    try:
        payload = json.loads(line_str, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as err:  # RecursionError: nesting deeper than the decoder
        raise InvalidRequestLineError(_error_response(None, -32700, "Parse error")) from err
    if isinstance(payload, list):
        msg = "Batch requests are not supported"
        raise InvalidRequestLineError(_error_response(None, -32600, msg))
    if not isinstance(payload, dict):
        raise InvalidRequestLineError(_error_response(None, -32600, "Invalid request"))
    req_id = payload.get("id")
    if isinstance(req_id, bool) or not isinstance(req_id, str | int | None):
        data = {"reason": "invalid_id"}
        raise InvalidRequestLineError(_error_response(None, -32600, "Invalid request", data))
    if payload.get("jsonrpc") != "2.0" or "method" not in payload:
        raise InvalidRequestLineError(_error_response(req_id, -32600, "Invalid request"))
    if "params" in payload and not isinstance(payload["params"], dict):
        data = {"field": "params"}
        msg = "params must be an object"
        raise InvalidRequestLineError(_error_response(req_id, -32602, msg, data))
    return payload


def _handle_request(
    stdout: TextIO, stderr: TextIO, router: Router, request: dict[str, Any]
) -> None:
    """Dispatches one validated request; fatal integrity errors terminate the process (exit 4)."""
    # JSON-RPC 2.0: only an *absent* id makes a notification; `"id": null` still gets a reply.
    is_notification = "id" not in request
    req_id = request.get("id")
    try:
        result = _dispatch_rpc_method(router, request["method"], request.get("params", {}))
        if not is_notification:
            _write_response(stdout, {"jsonrpc": "2.0", "id": req_id, "result": result})
    except _FATAL_ERRORS as err:
        _exit_fatal(stderr, router, err, phase="serve")
    except RpcError as err:
        if not is_notification:
            _write_response(stdout, _error_response(req_id, err.code, err.message, err.data))
        else:
            error_obj: dict[str, Any] = {"code": err.code, "message": err.message}
            if err.data is not None:
                error_obj["data"] = err.data
            _log_event(stderr, "error", "rpc_notification_error", error=error_obj)
    except Exception as err:
        _log_event(stderr, "error", "unhandled_error", error=str(err))
        if not is_notification:
            _write_response(stdout, _error_response(req_id, -32603, f"Internal error: {err}"))


class ShutdownCoordinator:
    """Turns SIGTERM/SIGINT into a shutdown that never interrupts a request (R4-MF-1).

    Python runs signal handlers between bytecodes on the main thread, so a handler that closes
    the router synchronously can snapshot a half-applied `report`. While a request is in flight
    the signal is only recorded and honoured once the request has been answered; while idle the
    handler closes immediately (a pure flag would hang in the blocking `readline`, PEP 475).
    """

    def __init__(self, router: Router, stderr: TextIO) -> None:
        self._router = router
        self._stderr = stderr
        self._in_flight = False
        self._requested = False

    def on_signal(self, sig: int, _frame: Any) -> None:
        _log_event(self._stderr, "info", "signal_received", signal=sig)
        if self._router.closing:
            return
        if self._in_flight:
            self._requested = True
            _log_event(self._stderr, "info", "shutdown_deferred", signal=sig)
            return
        self.close_and_exit()

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        """Marks a request as in flight; a shutdown requested meanwhile runs on exit."""
        self._in_flight = True
        try:
            yield
        finally:
            self._in_flight = False
            if self._requested and not self._router.closing:
                self.close_and_exit()

    def close_and_exit(self) -> NoReturn:
        sys.exit(self._router.close())


def serve(
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    router: Router,
    exit_on_eof: bool = True,
    shutdown: ShutdownCoordinator | None = None,
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
            error = _error_response(None, -32600, "Invalid request", {"reason": "line_too_long"})
            _write_response(stdout, error)
            continue
        if not line_str or not line_str.strip():
            continue
        try:
            request = _validate_request(line_str.strip())
        except InvalidRequestLineError as rejected:
            _write_response(stdout, rejected.response)
            continue
        if shutdown is None:
            _handle_request(stdout, stderr, router, request)
        else:
            with shutdown.in_flight():
                _handle_request(stdout, stderr, router, request)


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
    shutdown = ShutdownCoordinator(router, sys.stderr)
    signal.signal(signal.SIGTERM, shutdown.on_signal)
    signal.signal(signal.SIGINT, shutdown.on_signal)

    try:
        ready_params = router.boot()
    except _FATAL_ERRORS as err:
        _exit_fatal(sys.stderr, router, err, phase="boot")
    ready_notif = {"jsonrpc": "2.0", "method": "ready", "params": ready_params}
    sys.stdout.write(json.dumps(ready_notif) + "\n")
    sys.stdout.flush()

    serve(sys.stdin, sys.stdout, sys.stderr, router, shutdown=shutdown)


if __name__ == "__main__":
    main()
