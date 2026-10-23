"""Acceptance criteria of the end-to-end simulation: one source for the tests and the report.

Each `Check` states, in plain Spanish, what must hold, the threshold, and what was observed.
`tests/e2e/test_e2e_simulation.py` fails if any check fails; `scripts/e2e_report.py` prints them.
"""

from dataclasses import dataclass

from e2e import world
from e2e.scenarios import HostileResult, LearningResult, QuotaResult, RecoveryResult

LATENCY_CEILING_MS = 50.0
MIN_OPTIMAL_RATE = 0.85
MIN_TASK_OPTIMAL_RATE = 0.70
MIN_REWARD_VS_ORACLE = 0.95
MAX_REGRET_VS_FIRST_BLOCK = 0.50
MAX_REGRET_VS_RANDOM = 0.25


@dataclass(frozen=True)
class Check:
    """One verifiable statement: what must hold, the bar to clear, and what was measured."""

    statement: str
    criterion: str
    observed: str
    passed: bool


def _check(statement: str, criterion: str, observed: str, passed: bool) -> Check:
    return Check(statement, criterion, observed, bool(passed))


def learning_checks(result: LearningResult) -> list[Check]:
    first, last = result.blocks[0], result.blocks[-1]
    checks = [
        _check(
            "El router aprende: su error (regret) cae a menos de la mitad entre el primer y el "
            "último tramo",
            f"último tramo < {MAX_REGRET_VS_FIRST_BLOCK:.0%} del primero",
            f"{last['mean_regret']:.4f} frente a {first['mean_regret']:.4f}",
            last["mean_regret"] < MAX_REGRET_VS_FIRST_BLOCK * first["mean_regret"],
        ),
        _check(
            "Rinde muy por encima de repartir las peticiones al azar",
            f"error < {MAX_REGRET_VS_RANDOM:.0%} del error del azar",
            f"{last['mean_regret']:.4f} frente a {result.random_baseline_regret:.4f} (azar)",
            last["mean_regret"] < MAX_REGRET_VS_RANDOM * result.random_baseline_regret,
        ),
        _check(
            "Elige el mejor modelo en la gran mayoría de las peticiones (último tramo)",
            f"≥ {MIN_OPTIMAL_RATE:.0%}",
            f"{last['optimal_rate']:.1%}",
            last["optimal_rate"] >= MIN_OPTIMAL_RATE,
        ),
        _check(
            "La recompensa conseguida se acerca al máximo teórico (último tramo)",
            f"≥ {MIN_REWARD_VS_ORACLE:.0%} del óptimo",
            f"{result.router_mean_reward_last_block / result.oracle_mean_reward_last_block:.1%}",
            result.router_mean_reward_last_block
            >= MIN_REWARD_VS_ORACLE * result.oracle_mean_reward_last_block,
        ),
    ]
    checks += [
        _check(
            f"Aprende el mejor modelo para tareas de tipo «{task}»",
            f"≥ {MIN_TASK_OPTIMAL_RATE:.0%}",
            f"{result.per_task[task]['optimal_rate']:.1%}",
            result.per_task[task]["optimal_rate"] >= MIN_TASK_OPTIMAL_RATE,
        )
        for task in world.TASK_KINDS
    ]
    checks += [
        _check(
            f"Responde en milisegundos: «{method}» (percentil 99)",
            f"< {LATENCY_CEILING_MS:.0f} ms",
            f"{result.latency[method]['p99_ms']:.2f} ms",
            result.latency[method]["p99_ms"] < LATENCY_CEILING_MS,
        )
        for method in ("select", "report")
    ]
    return checks + _learning_lifecycle_checks(result)


def _learning_lifecycle_checks(result: LearningResult) -> list[Check]:
    logged = next(e for e in result.ope if e["candidate_alpha"] == result.alpha)
    return [
        _check(
            "Parada ordenada del servicio sin errores",
            "código de salida 0",
            f"código {result.graceful_exit_code}",
            result.graceful_exit_code == 0,
        ),
        _check(
            "Al reiniciar recupera todo lo aprendido",
            f"restaurado desde snapshot con {result.rounds} actualizaciones",
            f"snapshot={result.restart_restored_from_snapshot}, "
            f"actualizaciones={result.restart_state_version}",
            result.restart_restored_from_snapshot and result.restart_state_version == result.rounds,
        ),
        _check(
            "La evaluación offline (OPE) aprovecha todas las decisiones registradas",
            f"{result.rounds} decisiones en cada estimación",
            ", ".join(f"{int(e['n'])}" for e in result.ope),
            all(e["n"] == result.rounds and e["ess"] > 0.0 for e in result.ope),
        ),
        _check(
            "La evaluación offline de la política real coincide con la recompensa ganada",
            "diferencia < 0.000001",
            f"{logged['ips']:.6f} frente a {result.logged_mean_reward:.6f}",
            abs(logged["ips"] - result.logged_mean_reward) < 1e-6,
        ),
    ]


def quota_checks(result: QuotaResult) -> list[Check]:
    first, last = result.codex_share_by_block[0], result.codex_share_by_block[-1]
    return [
        _check(
            "El mejor modelo se frena solo antes de agotar su cuota (el precio sombra sube)",
            "cuota sin agotar y uso del último tramo < 1/3 del primero",
            f"{result.free_phase_codex_pulls:.0f} de {result.capacity_units:.0f} unidades; "
            f"uso {first:.0%} → {last:.0%}",
            result.free_phase_codex_pulls < result.capacity_units and first > 3 * last,
        ),
        _check(
            "El tráfico se reparte hacia el siguiente mejor modelo",
            "modelo alternativo = qwen3-coder",
            result.fallback_arm,
            result.fallback_arm == world.QWEN,
        ),
        _check(
            "La cuota nunca se supera, ni forzando el uso",
            f"exactamente {result.capacity_units:.0f} unidades consumidas",
            f"{result.total_codex_pulls} unidades (uso {result.final_utilization:.0%})",
            result.total_codex_pulls == result.capacity_units
            and abs(result.final_utilization - 1.0) < 1e-6,
        ),
        _check(
            "Con la cuota agotada rechaza la petición con un motivo claro",
            "error -32004 «quota_exhausted»",
            f"error {result.forced_error_code} «{result.forced_masked_reason}»",
            result.forced_error_code == -32004 and result.forced_masked_reason == "quota_exhausted",
        ),
    ]


def recovery_checks(result: RecoveryResult) -> list[Check]:
    checks: list[Check] = []
    for case in result.cases:
        checks += [
            _check(
                f"Caída brusca ({case.name}): no se pierde ninguna actualización confirmada",
                f"{case.acknowledged_updates} actualizaciones y mismo reparto por modelo",
                f"{case.state_version_after_restart} actualizaciones; "
                f"reparto idéntico={case.pulls_after_restart == case.pulls_before_crash}",
                case.state_version_after_restart == case.acknowledged_updates
                and case.pulls_after_restart == case.pulls_before_crash,
            ),
            _check(
                f"Caída brusca ({case.name}): las decisiones en vuelo se conservan y se pueden "
                "informar después",
                f"{case.pending_at_crash} decisiones pendientes, todas informadas con éxito",
                f"{case.pending_after_restart} pendientes tras reiniciar; "
                f"{case.pending_reports_applied} informadas; "
                f"versión final {case.state_version_after_pending_reports}",
                case.pending_after_restart == case.pending_at_crash
                and case.pending_reports_applied == case.pending_at_crash
                and case.state_version_after_pending_reports
                == case.acknowledged_updates + case.pending_at_crash,
            ),
        ]
    with_snapshot, without_snapshot = result.cases
    checks.append(
        _check(
            "Usa el snapshot si existe y recupera desde el registro si no existe",
            "snapshot restaurado solo en el primer caso",
            f"con snapshot={with_snapshot.restored_from_snapshot}, "
            f"sin snapshot={without_snapshot.restored_from_snapshot}",
            with_snapshot.restored_from_snapshot and not without_snapshot.restored_from_snapshot,
        )
    )
    return checks


def hostile_checks(result: HostileResult) -> list[Check]:
    checks = [
        _check(
            probe.name, f"respuesta {probe.expected}", f"respuesta {probe.observed}", probe.passed
        )
        for probe in result.probes
    ]
    checks.append(
        _check(
            "Tras todo el abuso el servicio sigue vivo y con el estado intacto",
            "activo, 1 actualización, 1 decisión pendiente",
            f"activo={result.daemon_alive_after}, actualizaciones={result.state_version_after}, "
            f"pendientes={result.pending_after}",
            result.daemon_alive_after
            and result.state_version_after == 1
            and result.pending_after == 1,
        )
    )
    return checks
