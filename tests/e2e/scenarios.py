"""End-to-end scenarios against the real daemon; each returns a plain-data result.

Tests assert on the results and `scripts/e2e_report.py` renders them, so a number reported to
the team is always one the test suite has checked.
"""

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from azuma_router.config import load_config, to_static_config
from azuma_router.ope import evaluate
from e2e import world
from e2e.client import DaemonProcess, RpcCallError
from e2e.driver import RoundRecord, report_params, route_and_report, running_daemon

LEARNING_ROUNDS = 3000
LEARNING_BLOCKS = 6
OPE_CANDIDATE_ALPHAS = (0.0, 0.3, 1.0)
UNLIMITED_QUOTA = 1e9
NO_CADENCE_SNAPSHOT = 10**9


# --------------------------------------------------------------------------- shared helpers


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def latency_summary(daemon: DaemonProcess) -> dict[str, dict[str, float]]:
    """p50/p95/p99 round-trip milliseconds per RPC method (client side, includes stdio)."""
    return {
        method: {
            "calls": float(len(samples)),
            "p50_ms": _percentile(samples, 50),
            "p95_ms": _percentile(samples, 95),
            "p99_ms": _percentile(samples, 99),
        }
        for method, samples in daemon.call_latencies_ms.items()
    }


def run_rounds(
    daemon: DaemonProcess,
    requests: list[world.SimRequest],
    env_rng: np.random.Generator,
    *,
    allowed_arms: list[str] | None = None,
) -> list[RoundRecord]:
    return [route_and_report(daemon, r, env_rng, allowed_arms=allowed_arms) for r in requests]


def arm_share(records: list[RoundRecord]) -> dict[str, float]:
    total = max(len(records), 1)
    return {arm: sum(r.chosen_arm == arm for r in records) / total for arm in world.ALL_ARM_IDS}


def optimal_rate(records: list[RoundRecord]) -> float:
    return float(np.mean([r.chosen_arm == r.oracle_arm for r in records])) if records else 0.0


# --------------------------------------------------------------------------- 1. learning


@dataclass
class LearningResult:
    rounds: int
    alpha: float
    blocks: list[dict[str, float]]
    random_baseline_regret: float
    per_task: dict[str, dict[str, Any]]
    oracle_mean_reward_last_block: float
    router_mean_reward_last_block: float
    latency: dict[str, dict[str, float]]
    throughput_rounds_per_s: float
    ope: list[dict[str, float]]
    logged_mean_reward: float
    graceful_exit_code: int
    restart_restored_from_snapshot: bool
    restart_state_version: int


def _block_summaries(records: list[RoundRecord]) -> list[dict[str, float]]:
    size = len(records) // LEARNING_BLOCKS
    summaries = []
    for index in range(LEARNING_BLOCKS):
        block = records[index * size : (index + 1) * size]
        summaries.append(
            {
                "start": float(index * size + 1),
                "end": float((index + 1) * size),
                "mean_regret": float(np.mean([r.regret for r in block])),
                "optimal_rate": optimal_rate(block),
            }
        )
    return summaries


def _per_task_summary(last_block: list[RoundRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task in world.TASK_KINDS:
        rows = [r for r in last_block if r.task_kind == task]
        summary[task] = {
            "requests": len(rows),
            "oracle_arm": rows[0].oracle_arm,
            "optimal_rate": optimal_rate(rows),
            "arm_share": arm_share(rows),
        }
    return summary


def _random_baseline_regret(requests: list[world.SimRequest]) -> float:
    """Expected regret of a router that picks an arm uniformly at random."""
    regrets = []
    for request in requests:
        oracle = world.expected_reward(world.best_arm(request), request)
        regrets.append(
            np.mean([oracle - world.expected_reward(a, request) for a in world.ALL_ARM_IDS])
        )
    return float(np.mean(regrets))


def _ope_estimates(work_dir: Path) -> list[dict[str, float]]:
    """Replays the decision log offline for candidate exploration strengths (FR-34)."""
    base = to_static_config(load_config(work_dir / "azuma.toml"))
    estimates = []
    for alpha in OPE_CANDIDATE_ALPHAS:
        result = evaluate(work_dir / "decisions.jsonl", dataclasses.replace(base, alpha=alpha))
        estimates.append({"candidate_alpha": alpha, **result})
    return estimates


def run_learning(
    work_dir: Path, rounds: int = LEARNING_ROUNDS, alpha: float = 0.3
) -> LearningResult:
    """Routes mixed traffic with unlimited quota; checks learning, speed, OPE and restart."""
    config = {"alpha": alpha, "codex_window_capacity": UNLIMITED_QUOTA}
    requests = world.make_traffic(np.random.default_rng(7), rounds)
    with running_daemon(work_dir, **config) as (daemon, _):
        records = run_rounds(daemon, requests, np.random.default_rng(8))
        latency = latency_summary(daemon)
        elapsed_s = sum(sum(v) for v in daemon.call_latencies_ms.values()) / 1000.0
        exit_code = daemon.terminate()
    with running_daemon(work_dir, **config) as (restarted, ready):
        restarted.terminate()
    last_block = records[-(rounds // LEARNING_BLOCKS) :]
    return LearningResult(
        rounds=rounds,
        alpha=alpha,
        blocks=_block_summaries(records),
        random_baseline_regret=_random_baseline_regret(requests),
        per_task=_per_task_summary(last_block),
        oracle_mean_reward_last_block=float(np.mean([r.oracle_reward for r in last_block])),
        router_mean_reward_last_block=float(np.mean([r.chosen_reward for r in last_block])),
        latency=latency,
        throughput_rounds_per_s=rounds / elapsed_s,
        ope=_ope_estimates(work_dir),
        logged_mean_reward=float(np.mean([r.observed_reward for r in records])),
        graceful_exit_code=exit_code,
        restart_restored_from_snapshot=bool(ready["restored_from_snapshot"]),
        restart_state_version=int(ready["state_version"]),
    )


# --------------------------------------------------------------------------- 2. quota

QUOTA_CAPACITY = 60.0
QUOTA_ROUNDS = 400
QUOTA_BLOCK = 50


@dataclass
class QuotaResult:
    capacity_units: float
    rounds: int
    free_phase_codex_pulls: int
    free_phase_codex_utilization: float
    codex_share_by_block: list[float]
    fallback_arm: str
    forced_extra_pulls: int
    total_codex_pulls: int
    final_utilization: float
    forced_error_code: int
    forced_masked_reason: str


def _codex_health(daemon: DaemonProcess) -> tuple[int, float]:
    health = daemon.call("health")["arms"][world.CODEX]
    return int(health["n_pulls"]), float(max(health["utilization"]))


def _drain_codex(
    daemon: DaemonProcess, requests: list[world.SimRequest], env_rng: np.random.Generator
) -> tuple[int, int, str]:
    """Forces codex-only traffic until the router refuses; returns (pulls, error code, reason)."""
    pulled = 0
    for request in requests:
        try:
            route_and_report(daemon, request, env_rng, allowed_arms=[world.CODEX])
        except RpcCallError as error:
            return pulled, error.code, str(error.data["masked_reason"][world.CODEX])
        pulled += 1
    return pulled, 0, "never_exhausted"


def _dominant_arm(records: list[RoundRecord]) -> str:
    shares = arm_share(records)
    return max(shares, key=lambda arm: shares[arm])


def run_quota(work_dir: Path) -> QuotaResult:
    """Only the best code arm (codex) has a small quota: it is throttled, then hard-capped."""
    requests = world.make_traffic(np.random.default_rng(11), QUOTA_ROUNDS, only="code_edit")
    env = np.random.default_rng(12)
    surplus = world.make_traffic(
        np.random.default_rng(13), int(QUOTA_CAPACITY) * 2, only="code_edit"
    )
    with running_daemon(work_dir, codex_window_capacity=QUOTA_CAPACITY) as (daemon, _):
        records = run_rounds(daemon, requests, env)
        free_pulls, free_utilization = _codex_health(daemon)
        extra, code, reason = _drain_codex(daemon, surplus, env)
        total_pulls, final_utilization = _codex_health(daemon)
        daemon.terminate()
    return QuotaResult(
        capacity_units=QUOTA_CAPACITY,
        rounds=QUOTA_ROUNDS,
        free_phase_codex_pulls=free_pulls,
        free_phase_codex_utilization=free_utilization,
        codex_share_by_block=[
            arm_share(records[i : i + QUOTA_BLOCK])[world.CODEX]
            for i in range(0, QUOTA_ROUNDS, QUOTA_BLOCK)
        ],
        fallback_arm=_dominant_arm([r for r in records if r.chosen_arm != world.CODEX]),
        forced_extra_pulls=extra,
        total_codex_pulls=total_pulls,
        final_utilization=final_utilization,
        forced_error_code=code,
        forced_masked_reason=reason,
    )


# --------------------------------------------------------------------------- 3. recovery

RECOVERY_BEFORE_SNAPSHOT = 200
RECOVERY_AFTER_SNAPSHOT = 100
RECOVERY_PENDING = 5
CRASH_WITHOUT_SNAPSHOT_ROUNDS = 150


@dataclass
class RecoveryCase:
    name: str
    acknowledged_updates: int
    pending_at_crash: int
    state_version_after_restart: int
    pending_after_restart: int
    restored_from_snapshot: bool
    pulls_before_crash: dict[str, int]
    pulls_after_restart: dict[str, int]
    pending_reports_applied: int
    state_version_after_pending_reports: int


@dataclass
class RecoveryResult:
    cases: list[RecoveryCase] = field(default_factory=list)


def _pulls(health: dict[str, Any]) -> dict[str, int]:
    return {arm: int(info["n_pulls"]) for arm, info in health["arms"].items()}


def _leave_pending(
    daemon: DaemonProcess, requests: list[world.SimRequest]
) -> list[tuple[str, world.SimRequest, str]]:
    pending = []
    for request in requests:
        selection = daemon.call("select", {"request": request.payload, "verbose": False})
        pending.append((selection["decision_id"], request, selection["arm"]["arm_id"]))
    return pending


RECOVERY_CONFIG = {
    "codex_window_capacity": UNLIMITED_QUOTA,
    "snapshot_every_updates": NO_CADENCE_SNAPSHOT,
}


def _run_until_crash(
    work_dir: Path,
    traffic: list[world.SimRequest],
    env: np.random.Generator,
    *,
    snapshot_after: int | None,
    rounds: int,
) -> tuple[list[tuple[str, world.SimRequest, str]], dict[str, int]]:
    """Serves `rounds` acknowledged rounds, leaves the rest pending, then SIGKILLs the daemon."""
    with running_daemon(work_dir, **RECOVERY_CONFIG) as (daemon, _):
        if snapshot_after is not None:
            run_rounds(daemon, traffic[:snapshot_after], env)
            daemon.call("snapshot")
            run_rounds(daemon, traffic[snapshot_after:rounds], env)
        else:
            run_rounds(daemon, traffic[:rounds], env)
        left_pending = _leave_pending(daemon, traffic[rounds:])
        pulls_before = _pulls(daemon.call("health"))
        daemon.kill()
    return left_pending, pulls_before


def _crash_and_recover(
    work_dir: Path, name: str, *, snapshot_after: int | None, rounds: int, pending: int
) -> RecoveryCase:
    traffic = world.make_traffic(np.random.default_rng(21), rounds + pending)
    env = np.random.default_rng(22)
    left_pending, pulls_before = _run_until_crash(
        work_dir, traffic, env, snapshot_after=snapshot_after, rounds=rounds
    )
    with running_daemon(work_dir, **RECOVERY_CONFIG) as (revived, ready):
        pulls_after = _pulls(revived.call("health"))
        for decision_id, request, arm in left_pending:
            revived.call("report", report_params(decision_id, world.observe(arm, request, env)))
        final = revived.call("health")
        revived.terminate()
    return RecoveryCase(
        name=name,
        acknowledged_updates=rounds,
        pending_at_crash=pending,
        state_version_after_restart=int(ready["state_version"]),
        pending_after_restart=int(ready["pending_replayed"]),
        restored_from_snapshot=bool(ready["restored_from_snapshot"]),
        pulls_before_crash=pulls_before,
        pulls_after_restart=pulls_after,
        pending_reports_applied=pending - int(final["pending"]),
        state_version_after_pending_reports=int(final["state_version"]),
    )


def run_recovery(work_dir: Path) -> RecoveryResult:
    """SIGKILL the daemon mid-traffic (with and without a snapshot) and check nothing is lost."""
    with_snapshot = _crash_and_recover(
        work_dir / "with-snapshot",
        "con snapshot previo",
        snapshot_after=RECOVERY_BEFORE_SNAPSHOT,
        rounds=RECOVERY_BEFORE_SNAPSHOT + RECOVERY_AFTER_SNAPSHOT,
        pending=RECOVERY_PENDING,
    )
    without_snapshot = _crash_and_recover(
        work_dir / "without-snapshot",
        "sin snapshot previo",
        snapshot_after=None,
        rounds=CRASH_WITHOUT_SNAPSHOT_ROUNDS,
        pending=RECOVERY_PENDING,
    )
    return RecoveryResult(cases=[with_snapshot, without_snapshot])


# --------------------------------------------------------------------------- 4. hostile input

LINE_LIMIT_BYTES = 1_048_576
HEALTH_PROBE_ID = 99


@dataclass
class Probe:
    name: str
    expected: str
    observed: str

    @property
    def passed(self) -> bool:
        return self.expected == self.observed


@dataclass
class HostileResult:
    probes: list[Probe]
    daemon_alive_after: bool
    state_version_after: int
    pending_after: int


def _code_of(response: dict[str, Any]) -> str:
    if "error" in response:
        return str(response["error"]["code"])
    result = response["result"]
    return "duplicate" if result.get("reason") == "duplicate" else "ok"


def served_health(response: dict[str, Any]) -> str:
    """ "ok" only if `response` is the ready answer to the health request behind a probe."""
    result = response.get("result")
    if (
        response.get("id") == HEALTH_PROBE_ID
        and isinstance(result, dict)
        and result.get("ready") is True
    ):
        return "ok"
    return f"respuesta inesperada: {json.dumps(response)[:80]}"


def _rpc(method: str, params: Any, request_id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def _first_selection(daemon: DaemonProcess, request: world.SimRequest, arms: list[str]) -> str:
    params = {"request": request.payload, "allowed_arms": arms, "verbose": False}
    return str(daemon.call("select", params)["decision_id"])


def _oversized_line_probes(daemon: DaemonProcess) -> list[Probe]:
    """Lines over 1 MiB are rejected and the request right behind each must still be served."""
    probes = []
    for label, content_bytes in (
        ("un byte por encima del límite", LINE_LIMIT_BYTES),
        ("el doble del límite", LINE_LIMIT_BYTES * 2),
    ):
        daemon.write_raw("x" * content_bytes + "\n" + _rpc("health", {}, HEALTH_PROBE_ID) + "\n")
        rejected = daemon.read_response()
        next_request = daemon.read_response()
        probes += [
            Probe(f"Línea de más de 1 MiB ({label}): rechazada", "-32600", _code_of(rejected)),
            Probe(
                f"Línea de más de 1 MiB ({label}): la petición siguiente se atiende",
                "ok",
                served_health(next_request),
            ),
        ]
    return probes


def _protocol_probes(daemon: DaemonProcess, request: world.SimRequest) -> list[Probe]:
    bad_context = {**request.payload, "task_kind": "make_coffee"}
    negative_tokens = {**request.payload, "input_tokens": -5}
    cases: list[tuple[str, str, str]] = [
        ("JSON malformado", "-32700", "{not json"),
        (
            "Valor NaN en el mensaje",
            "-32700",
            '{"jsonrpc":"2.0","id":1,"method":"health","params":NaN}',
        ),
        ("Petición por lotes (no soportada)", "-32600", "[]"),
        (
            "Parámetros que no son un objeto",
            "-32602",
            '{"jsonrpc":"2.0","id":1,"method":"health","params":[]}',
        ),
        ("Método inexistente", "-32601", _rpc("format_disk", {})),
        ("Tipo de tarea desconocido", "-32001", _rpc("select", {"request": bad_context})),
        ("Número de tokens negativo", "-32001", _rpc("select", {"request": negative_tokens})),
        (
            "Modelo desconocido en la lista de permitidos",
            "-32602",
            _rpc("select", {"request": request.payload, "allowed_arms": ["nope"]}),
        ),
        (
            "Informe de una decisión que no existe",
            "-32002",
            _rpc("report", {"decision_id": "ghost", "quality": 0.5}),
        ),
    ]
    return [Probe(name, code, _code_of(daemon.call_raw(line))) for name, code, line in cases]


def _report_validation_probes(daemon: DaemonProcess, request: world.SimRequest) -> list[Probe]:
    codex_decision = _first_selection(daemon, request, [world.CODEX])
    haiku_decision = _first_selection(daemon, request, [world.HAIKU])
    good = {"quality": 0.8, "latency_s": 3.0}
    cases: list[tuple[str, str, str]] = [
        (
            "Calidad fuera de rango (1.5)",
            "-32602",
            _rpc("report", {"decision_id": haiku_decision, "quality": 1.5}),
        ),
        (
            "Coste informado en un modelo de suscripción",
            "-32602",
            _rpc("report", {"decision_id": codex_decision, "quality": 0.8, "cost_usd": 0.5}),
        ),
        ("Informe válido", "ok", _rpc("report", {"decision_id": haiku_decision, **good})),
        (
            "Mismo informe enviado dos veces",
            "duplicate",
            _rpc("report", {"decision_id": haiku_decision, **good}),
        ),
    ]
    return [Probe(name, code, _code_of(daemon.call_raw(line))) for name, code, line in cases]


def run_hostile_input(work_dir: Path) -> HostileResult:
    """Throws malformed, invalid and abusive input at the daemon; it must keep serving."""
    request = world.make_traffic(np.random.default_rng(31), 1, only="chat")[0]
    with running_daemon(work_dir, codex_window_capacity=UNLIMITED_QUOTA) as (daemon, _):
        probes = _protocol_probes(daemon, request)
        probes += _report_validation_probes(daemon, request)
        probes += _oversized_line_probes(daemon)
        health = daemon.call("health")
        daemon.terminate()
    return HostileResult(
        probes=probes,
        daemon_alive_after=bool(health["ready"]),
        state_version_after=int(health["state_version"]),
        pending_after=int(health["pending"]),
    )


# --------------------------------------------------------------------------- serialization


def to_plain(result: Any) -> dict[str, Any]:
    """Dataclass -> JSON-ready dict (probes get their computed `passed` flag)."""
    data = asdict(result)
    for probe, original in zip(data.get("probes", []), getattr(result, "probes", []), strict=True):
        probe["passed"] = original.passed
    return data
