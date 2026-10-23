"""Negative and boundary cases of the shared acceptance criteria (SC-1, SC-2 of the review).

A criterion that nobody ever sees fail could be relaxed by accident and keep both the e2e test and
the report green, so each one is driven past its bar with synthetic results (no daemon).
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from e2e import checks, scenarios, world
from e2e.test_report import make_hostile, make_learning, make_quota, make_recovery_case


def failed_statements(group: list[checks.Check]) -> list[str]:
    return [check.statement for check in group if not check.passed]


def with_last_block(result: scenarios.LearningResult, **changes: float) -> scenarios.LearningResult:
    blocks = [*result.blocks[:-1], {**result.blocks[-1], **changes}]
    return dataclasses.replace(result, blocks=blocks)


class TestLearningChecks:
    def test_a_healthy_run_passes_everything(self) -> None:
        assert failed_statements(checks.learning_checks(make_learning())) == []

    def test_regret_that_only_halves_exactly_does_not_count_as_learning(self) -> None:
        learning = make_learning()
        first = learning.blocks[0]["mean_regret"]
        broken = with_last_block(learning, mean_regret=checks.MAX_REGRET_VS_FIRST_BLOCK * first)
        assert any("aprende" in s for s in failed_statements(checks.learning_checks(broken)))

    def test_regret_not_clearly_better_than_random_fails(self) -> None:
        learning = make_learning()
        broken = dataclasses.replace(
            learning,
            random_baseline_regret=learning.blocks[-1]["mean_regret"] / checks.MAX_REGRET_VS_RANDOM,
        )
        assert any("al azar" in s for s in failed_statements(checks.learning_checks(broken)))

    def test_optimal_rate_is_inclusive_at_the_bar_and_fails_just_below_it(self) -> None:
        at_bar = with_last_block(make_learning(), optimal_rate=checks.MIN_OPTIMAL_RATE)
        assert failed_statements(checks.learning_checks(at_bar)) == []
        below = with_last_block(make_learning(), optimal_rate=checks.MIN_OPTIMAL_RATE - 0.001)
        assert any("mejor modelo" in s for s in failed_statements(checks.learning_checks(below)))

    def test_reward_below_the_oracle_bar_fails(self) -> None:
        learning = make_learning()
        broken = dataclasses.replace(
            learning,
            router_mean_reward_last_block=learning.oracle_mean_reward_last_block
            * (checks.MIN_REWARD_VS_ORACLE - 0.01),
        )
        assert any("recompensa" in s for s in failed_statements(checks.learning_checks(broken)))

    @pytest.mark.parametrize("method", ["select", "report"])
    def test_p99_latency_at_the_ceiling_fails(self, method: str) -> None:
        learning = make_learning()
        latency = {
            **learning.latency,
            method: {**learning.latency[method], "p99_ms": checks.LATENCY_CEILING_MS},
        }
        broken = dataclasses.replace(learning, latency=latency)
        assert failed_statements(checks.learning_checks(broken)) == [
            f"Responde en milisegundos: «{method}» (percentil 99)"
        ]

    def test_a_task_kind_below_its_bar_fails_alone(self) -> None:
        learning = make_learning()
        task = world.TASK_KINDS[0]
        per_task = {
            **learning.per_task,
            task: {**learning.per_task[task], "optimal_rate": checks.MIN_TASK_OPTIMAL_RATE - 0.01},
        }
        broken = dataclasses.replace(learning, per_task=per_task)
        assert failed_statements(checks.learning_checks(broken)) == [
            f"Aprende el mejor modelo para tareas de tipo «{task}»"
        ]

    @pytest.mark.parametrize(
        "change",
        [
            {"graceful_exit_code": 1},
            {"restart_restored_from_snapshot": False},
            {"restart_state_version": 2999},
            {"logged_mean_reward": 0.5},
        ],
    )
    def test_lifecycle_failures_are_reported(self, change: dict[str, Any]) -> None:
        broken = dataclasses.replace(make_learning(), **change)
        assert failed_statements(checks.learning_checks(broken)) != []

    def test_ope_estimate_that_misses_a_decision_fails(self) -> None:
        learning = make_learning()
        ope = [{**learning.ope[0], "n": learning.rounds - 1.0}, *learning.ope[1:]]
        broken = dataclasses.replace(learning, ope=ope)
        assert any("OPE" in s for s in failed_statements(checks.learning_checks(broken)))


class TestQuotaChecks:
    def test_a_healthy_run_passes_everything(self) -> None:
        assert failed_statements(checks.quota_checks(make_quota())) == []

    @pytest.mark.parametrize(
        "change",
        [
            {"free_phase_codex_pulls": 60},
            {"codex_share_by_block": [0.4, 0.3, 0.2, 0.2]},
            {"fallback_arm": world.HAIKU},
            {"total_codex_pulls": 59},
            {"final_utilization": 0.9},
            {"forced_error_code": -32603},
            {"forced_masked_reason": "other"},
        ],
    )
    def test_each_quota_violation_is_reported(self, change: dict[str, Any]) -> None:
        broken = dataclasses.replace(make_quota(), **change)
        assert failed_statements(checks.quota_checks(broken)) != []


def recovery_with(
    mutate: Callable[[scenarios.RecoveryCase], scenarios.RecoveryCase],
) -> scenarios.RecoveryResult:
    return scenarios.RecoveryResult(
        cases=[
            mutate(make_recovery_case("con snapshot previo", from_snapshot=True)),
            make_recovery_case("sin snapshot previo", from_snapshot=False),
        ]
    )


class TestRecoveryChecks:
    def test_a_healthy_run_passes_everything(self) -> None:
        assert failed_statements(checks.recovery_checks(recovery_with(lambda c: c))) == []

    @pytest.mark.parametrize(
        "change",
        [
            {"state_version_after_restart": 299},
            {"pulls_after_restart": {world.CODEX: 99, world.QWEN: 101, world.HAIKU: 100}},
            {"pending_after_restart": 4},
            {"pending_reports_applied": 4},
            {"state_version_after_pending_reports": 304},
        ],
    )
    def test_losing_acknowledged_or_in_flight_work_is_reported(
        self, change: dict[str, Any]
    ) -> None:
        result = recovery_with(lambda case: dataclasses.replace(case, **change))
        assert len(failed_statements(checks.recovery_checks(result))) == 1

    def test_restoring_from_snapshot_in_the_wrong_case_is_reported(self) -> None:
        result = recovery_with(lambda case: dataclasses.replace(case, restored_from_snapshot=False))
        assert failed_statements(checks.recovery_checks(result)) == [
            "Usa el snapshot si existe y recupera desde el registro si no existe"
        ]


class TestHostileChecks:
    def test_a_healthy_run_passes_everything(self) -> None:
        assert failed_statements(checks.hostile_checks(make_hostile())) == []

    def test_an_unexpected_response_to_a_probe_fails_that_probe(self) -> None:
        hostile = make_hostile()
        wrong = scenarios.Probe("Línea de más de 1 MiB (x)", "-32600", "ok")
        broken = dataclasses.replace(hostile, probes=[wrong])
        assert failed_statements(checks.hostile_checks(broken)) == [wrong.name]

    @pytest.mark.parametrize(
        "change",
        [{"daemon_alive_after": False}, {"state_version_after": 2}, {"pending_after": 0}],
    )
    def test_a_daemon_that_died_or_changed_state_fails(self, change: dict[str, Any]) -> None:
        broken = dataclasses.replace(make_hostile(), **change)
        assert failed_statements(checks.hostile_checks(broken)) == [
            "Tras todo el abuso el servicio sigue vivo y con el estado intacto"
        ]


class TestFramingProbeOutcome:
    """The request behind an oversized line must be answered *as that request* (SC-1)."""

    def test_the_health_answer_with_the_probe_id_is_served(self) -> None:
        response = {"jsonrpc": "2.0", "id": scenarios.HEALTH_PROBE_ID, "result": {"ready": True}}
        assert scenarios.served_health(response) == "ok"

    @pytest.mark.parametrize(
        "response",
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"ready": True}},
            {"jsonrpc": "2.0", "id": scenarios.HEALTH_PROBE_ID, "result": {"ready": False}},
            {"jsonrpc": "2.0", "id": scenarios.HEALTH_PROBE_ID, "result": {}},
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "x"}},
        ],
    )
    def test_any_other_answer_is_not_a_served_request(self, response: dict[str, Any]) -> None:
        assert scenarios.served_health(response) != "ok"
