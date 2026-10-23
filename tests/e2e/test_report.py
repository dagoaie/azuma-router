"""Unit tests for the sprint-delivery report renderer (synthetic results, no daemon)."""

import dataclasses
import json
from pathlib import Path

from e2e import report, scenarios, world
from e2e.checks import Check

META = report.RunMeta(
    date="2026-10-23", branch="main", commit="abc1234", dirty=False, version="0.1.0"
)


def make_learning() -> scenarios.LearningResult:
    blocks = [
        {"start": 1.0 + i * 500, "end": (i + 1) * 500.0, "mean_regret": r, "optimal_rate": o}
        for i, (r, o) in enumerate(
            [(0.05, 0.74), (0.03, 0.82), (0.02, 0.88), (0.015, 0.9), (0.014, 0.91), (0.013, 0.92)]
        )
    ]
    per_task = {
        task: {
            "requests": 100,
            "oracle_arm": world.CODEX,
            "optimal_rate": 0.9,
            "arm_share": {world.CODEX: 0.9},
        }
        for task in world.TASK_KINDS
    }
    latency = {
        m: {"calls": 3000.0, "p50_ms": 0.4, "p95_ms": 0.6, "p99_ms": 1.2}
        for m in ("select", "report")
    }
    ope = [
        {"candidate_alpha": a, "ips": 0.79, "dr": 0.79, "n": 3000.0, "ess": 2500.0}
        for a in (0.0, 0.3, 1.0)
    ]
    return scenarios.LearningResult(
        rounds=3000,
        alpha=0.3,
        blocks=blocks,
        random_baseline_regret=0.148,
        per_task=per_task,
        oracle_mean_reward_last_block=0.81,
        router_mean_reward_last_block=0.80,
        latency=latency,
        throughput_rounds_per_s=1700.0,
        ope=ope,
        logged_mean_reward=0.79,
        graceful_exit_code=0,
        restart_restored_from_snapshot=True,
        restart_state_version=3000,
    )


def make_quota() -> scenarios.QuotaResult:
    return scenarios.QuotaResult(
        capacity_units=60.0,
        rounds=400,
        free_phase_codex_pulls=37,
        free_phase_codex_utilization=0.6,
        codex_share_by_block=[0.42, 0.3, 0.2, 0.1, 0.08, 0.05, 0.04, 0.04],
        fallback_arm=world.QWEN,
        forced_extra_pulls=23,
        total_codex_pulls=60,
        final_utilization=1.0,
        forced_error_code=-32004,
        forced_masked_reason="quota_exhausted",
    )


def make_recovery_case(name: str, *, from_snapshot: bool) -> scenarios.RecoveryCase:
    pulls = {world.CODEX: 100, world.QWEN: 100, world.HAIKU: 100}
    return scenarios.RecoveryCase(
        name=name,
        acknowledged_updates=300,
        pending_at_crash=5,
        state_version_after_restart=300,
        pending_after_restart=5,
        restored_from_snapshot=from_snapshot,
        pulls_before_crash=pulls,
        pulls_after_restart=dict(pulls),
        pending_reports_applied=5,
        state_version_after_pending_reports=305,
    )


def make_hostile() -> scenarios.HostileResult:
    probes = [scenarios.Probe("Línea de más de 1 MiB (x)", "-32600", "-32600")]
    return scenarios.HostileResult(
        probes=probes, daemon_alive_after=True, state_version_after=1, pending_after=1
    )


def make_results() -> report.Results:
    return report.Results(
        learning=make_learning(),
        quota=make_quota(),
        recovery=scenarios.RecoveryResult(
            cases=[
                make_recovery_case("con snapshot previo", from_snapshot=True),
                make_recovery_case("sin snapshot previo", from_snapshot=False),
            ]
        ),
        hostile=make_hostile(),
    )


class TestMarkdownHelpers:
    def test_table_escapes_pipes_inside_cells(self) -> None:
        rendered = report.table(["a", "b"], [["x|y", "z"]])
        assert rendered.splitlines()[2] == "| x\\|y | z |"

    def test_table_has_header_separator_and_one_line_per_row(self) -> None:
        lines = report.table(["a", "b"], [["1", "2"], ["3", "4"]]).splitlines()
        assert lines[1] == "|---|---|"
        assert len(lines) == 4

    def test_verdict_counts_passed_checks(self) -> None:
        group = [Check("s", "c", "o", True), Check("s", "c", "o", False)]
        assert report.verdict(group) == "❌ CON FALLOS (1/2 comprobaciones)"

    def test_change_label_respects_direction_and_tolerance(self) -> None:
        assert report.change_label(True, 0.80, 0.90) == "mejora"
        assert report.change_label(False, 0.80, 0.90) == "empeora"
        assert report.change_label(False, 1.00, 1.01) == "sin cambio"
        assert report.change_label(True, 0.0, 5.0) == "sin cambio"


class TestRenderReport:
    def test_a_passing_run_is_declared_approved_with_every_section(self) -> None:
        text = report.render_report(META, make_results(), history=[])
        assert "✅ APROBADO" in text
        assert "❌" not in text
        for title in report.SCENARIO_TITLES.values():
            assert title in text
        for heading in ("## Evolución", "## Alcance y limitaciones", "## Glosario"):
            assert heading in text

    def test_regret_curve_is_a_mermaid_chart_with_one_point_per_block(self) -> None:
        text = report.render_report(META, make_results(), history=[])
        chart = text.split("```mermaid\n", 1)[1].split("```", 1)[0]
        assert chart.startswith("xychart-beta")
        line = next(row for row in chart.splitlines() if "line [0.0500" in row)
        assert line.count(",") == scenarios.LEARNING_BLOCKS - 1

    def test_a_failing_check_shows_up_in_the_report(self) -> None:
        results = make_results()
        broken = dataclasses.replace(results.quota, total_codex_pulls=61)
        text = report.render_report(META, dataclasses.replace(results, quota=broken), [])
        assert "❌ HAY COMPROBACIONES FALLIDAS" in text
        assert "| ❌ |" in text

    def test_the_first_report_says_there_is_nothing_to_compare_with(self) -> None:
        text = report.render_report(META, make_results(), history=[])
        assert "Es el primer informe" in text
        assert "Aún no hay un informe anterior" in text

    def test_later_reports_show_the_change_against_the_previous_one(self) -> None:
        previous = report.HistoryEntry(
            "2026-10-01",
            "old0001",
            30,
            31,
            {
                "final_regret": 0.03,
                "optimal_rate": 0.8,
                "reward_ratio": 0.9,
                "select_p99_ms": 1.0,
                "report_p99_ms": 0.5,
                "throughput": 1500.0,
            },
        )
        text = report.render_report(META, make_results(), history=[previous])
        assert "Cambios respecto al informe del 2026-10-01" in text
        assert "mejora" in text
        assert "`old0001`" in text

    def test_dirty_tree_is_disclosed_in_the_header(self) -> None:
        text = report.render_report(dataclasses.replace(META, dirty=True), make_results(), [])
        assert "con cambios sin confirmar" in text


class TestPersistence:
    def test_write_report_stores_markdown_and_results_json(self, tmp_path: Path) -> None:
        path = report.write_report(tmp_path, META, make_results())
        assert path == tmp_path / "2026-10-23-e2e-report.md"
        data = json.loads((tmp_path / "2026-10-23-results.json").read_text(encoding="utf-8"))
        assert data["checks_passed"] == data["checks_total"]
        assert set(data["scenarios"]) == {"learning", "quota", "recovery", "hostile"}

    def test_history_returns_earlier_runs_oldest_first_and_skips_the_current_date(
        self, tmp_path: Path
    ) -> None:
        for day in ("2026-10-10", "2026-10-01", "2026-10-23"):
            report.write_report(tmp_path, dataclasses.replace(META, date=day), make_results())
        history = report.load_history(tmp_path, exclude_date="2026-10-23")
        assert [entry.date for entry in history] == ["2026-10-01", "2026-10-10"]

    def test_rewriting_the_same_day_keeps_a_single_history_row(self, tmp_path: Path) -> None:
        report.write_report(tmp_path, META, make_results())
        path = report.write_report(tmp_path, META, make_results())
        rows = [
            ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("| 2026-")
        ]
        assert len(rows) == 1
