"""End-to-end simulation: the real daemon, over stdio, against a simulated world.

Excluded from the default run (`-m e2e` selects it). The acceptance criteria live in
`e2e.checks` and the same scenarios feed `scripts/e2e_report.py`, so every figure in a delivery
report is a figure asserted here.
"""

import pytest

from e2e import checks, scenarios, world

pytestmark = pytest.mark.e2e


def failed(results: list[checks.Check]) -> list[str]:
    return [
        f"{c.statement} | exigido: {c.criterion} | observado: {c.observed}"
        for c in results
        if not c.passed
    ]


@pytest.fixture(scope="module")
def learning(tmp_path_factory: pytest.TempPathFactory) -> scenarios.LearningResult:
    return scenarios.run_learning(tmp_path_factory.mktemp("learning"))


@pytest.fixture(scope="module")
def quota(tmp_path_factory: pytest.TempPathFactory) -> scenarios.QuotaResult:
    return scenarios.run_quota(tmp_path_factory.mktemp("quota"))


@pytest.fixture(scope="module")
def recovery(tmp_path_factory: pytest.TempPathFactory) -> scenarios.RecoveryResult:
    return scenarios.run_recovery(tmp_path_factory.mktemp("recovery"))


@pytest.fixture(scope="module")
def hostile(tmp_path_factory: pytest.TempPathFactory) -> scenarios.HostileResult:
    return scenarios.run_hostile_input(tmp_path_factory.mktemp("hostile"))


class TestLearning:
    def test_all_acceptance_checks_pass(self, learning: scenarios.LearningResult) -> None:
        assert failed(checks.learning_checks(learning)) == []

    def test_checks_cover_every_task_kind_and_both_rpc_methods(
        self, learning: scenarios.LearningResult
    ) -> None:
        statements = [c.statement for c in checks.learning_checks(learning)]
        for task in world.TASK_KINDS:
            assert any(f"«{task}»" in s for s in statements)
        assert sum("Responde en milisegundos" in s for s in statements) == 2

    def test_per_task_arm_share_of_the_oracle_equals_its_optimal_rate(
        self, learning: scenarios.LearningResult
    ) -> None:
        for summary in learning.per_task.values():
            assert summary["arm_share"][summary["oracle_arm"]] == summary["optimal_rate"]


class TestQuota:
    def test_all_acceptance_checks_pass(self, quota: scenarios.QuotaResult) -> None:
        assert failed(checks.quota_checks(quota)) == []


class TestRecovery:
    def test_all_acceptance_checks_pass(self, recovery: scenarios.RecoveryResult) -> None:
        assert failed(checks.recovery_checks(recovery)) == []

    def test_both_crash_cases_are_exercised(self, recovery: scenarios.RecoveryResult) -> None:
        assert [case.restored_from_snapshot for case in recovery.cases] == [True, False]


class TestHostileInput:
    def test_all_acceptance_checks_pass(self, hostile: scenarios.HostileResult) -> None:
        assert failed(checks.hostile_checks(hostile)) == []

    def test_oversized_lines_are_probed(self, hostile: scenarios.HostileResult) -> None:
        oversized = [p for p in hostile.probes if "1 MiB" in p.name]
        assert len(oversized) == 4
