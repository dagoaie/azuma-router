"""Sprint-delivery report: renders the e2e scenario results as Markdown for client and PM.

Pure rendering plus a small CLI (`main`), driven by `scripts/e2e_report.py`. Each run writes a
`<date>-e2e-report.md` and a `<date>-results.json`; the JSON files of earlier runs feed the
"evolution" section, so the report shows how the tool changes from sprint to sprint.
"""

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from e2e import checks, scenarios, world
from e2e.checks import Check

DEFAULT_OUTPUT_DIR = Path("docs/reports/e2e")
UNCHANGED_TOLERANCE = 0.02

ARM_NAMES = {
    world.CODEX: "gpt-5-codex (suscripción con cuota)",
    world.QWEN: "qwen3-coder (pago por uso)",
    world.HAIKU: "claude-haiku (pago por uso)",
}
TASK_NAMES = {
    "code_edit": "Editar código",
    "planning": "Planificar",
    "summarize": "Resumir",
    "extract": "Extraer datos",
    "chat": "Conversar",
}
SCENARIO_TITLES = {
    "learning": "1. Aprendizaje y rendimiento",
    "quota": "2. Gestión de la cuota",
    "recovery": "3. Recuperación ante caídas",
    "hostile": "4. Entradas hostiles",
}

# key, label, format, higher_is_better
METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("final_regret", "Error medio al final del aprendizaje", "{:.4f}", False),
    ("optimal_rate", "Aciertos del mejor modelo (último tramo)", "{:.1%}", True),
    ("reward_ratio", "Recompensa respecto al máximo teórico", "{:.1%}", True),
    ("select_p99_ms", "Tiempo de decisión, percentil 99 (ms)", "{:.2f}", False),
    ("report_p99_ms", "Tiempo de registro del resultado, percentil 99 (ms)", "{:.2f}", False),
    ("throughput", "Peticiones por segundo", "{:,.0f}", True),
)


@dataclass(frozen=True)
class RunMeta:
    date: str
    branch: str
    commit: str
    dirty: bool
    version: str


@dataclass(frozen=True)
class Results:
    learning: scenarios.LearningResult
    quota: scenarios.QuotaResult
    recovery: scenarios.RecoveryResult
    hostile: scenarios.HostileResult


@dataclass(frozen=True)
class HistoryEntry:
    date: str
    commit: str
    checks_passed: int
    checks_total: int
    metrics: dict[str, float]


# --------------------------------------------------------------------------- derived data


def run_all(work_dir: Path) -> Results:
    return Results(
        learning=scenarios.run_learning(work_dir / "learning"),
        quota=scenarios.run_quota(work_dir / "quota"),
        recovery=scenarios.run_recovery(work_dir / "recovery"),
        hostile=scenarios.run_hostile_input(work_dir / "hostile"),
    )


def all_checks(results: Results) -> dict[str, list[Check]]:
    return {
        "learning": checks.learning_checks(results.learning),
        "quota": checks.quota_checks(results.quota),
        "recovery": checks.recovery_checks(results.recovery),
        "hostile": checks.hostile_checks(results.hostile),
    }


def count_passed(checks_by_scenario: dict[str, list[Check]]) -> tuple[int, int]:
    flat = [c for group in checks_by_scenario.values() for c in group]
    return sum(c.passed for c in flat), len(flat)


def key_metrics(learning: scenarios.LearningResult) -> dict[str, float]:
    last = learning.blocks[-1]
    return {
        "final_regret": last["mean_regret"],
        "optimal_rate": last["optimal_rate"],
        "reward_ratio": learning.router_mean_reward_last_block
        / learning.oracle_mean_reward_last_block,
        "select_p99_ms": learning.latency["select"]["p99_ms"],
        "report_p99_ms": learning.latency["report"]["p99_ms"],
        "throughput": learning.throughput_rounds_per_s,
    }


def collect_meta(today: str) -> RunMeta:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    return RunMeta(
        date=today,
        branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        commit=git("rev-parse", "--short", "HEAD"),
        dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        version=version("azuma-router"),
    )


# --------------------------------------------------------------------------- history


def results_path(output_dir: Path, report_date: str) -> Path:
    return output_dir / f"{report_date}-results.json"


def report_path(output_dir: Path, report_date: str) -> Path:
    return output_dir / f"{report_date}-e2e-report.md"


def load_history(output_dir: Path, exclude_date: str) -> list[HistoryEntry]:
    """Earlier runs, oldest first; the run being rendered (same date) is left out."""
    entries = []
    for path in sorted(output_dir.glob("*-results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["meta"]["date"] == exclude_date:
            continue
        entries.append(
            HistoryEntry(
                date=data["meta"]["date"],
                commit=data["meta"]["commit"],
                checks_passed=data["checks_passed"],
                checks_total=data["checks_total"],
                metrics=data["metrics"],
            )
        )
    return entries


def results_json(meta: RunMeta, results: Results) -> dict[str, Any]:
    checks_by_scenario = all_checks(results)
    passed, total = count_passed(checks_by_scenario)
    return {
        "meta": asdict(meta),
        "checks_passed": passed,
        "checks_total": total,
        "metrics": key_metrics(results.learning),
        "checks": {name: [asdict(c) for c in group] for name, group in checks_by_scenario.items()},
        "scenarios": {
            "learning": scenarios.to_plain(results.learning),
            "quota": scenarios.to_plain(results.quota),
            "recovery": scenarios.to_plain(results.recovery),
            "hostile": scenarios.to_plain(results.hostile),
        },
    }


# --------------------------------------------------------------------------- markdown helpers


def table(header: list[str], rows: list[list[str]]) -> str:
    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def checks_table(group: list[Check]) -> str:
    rows = [["✅" if c.passed else "❌", c.statement, c.criterion, c.observed] for c in group]
    return table(["", "Qué se comprueba", "Exigido", "Observado"], rows)


def verdict(group: list[Check]) -> str:
    passed = sum(c.passed for c in group)
    mark = "✅ APROBADO" if passed == len(group) else "❌ CON FALLOS"
    return f"{mark} ({passed}/{len(group)} comprobaciones)"


def change_label(key_higher_is_better: bool, previous: float, current: float) -> str:
    if previous == 0.0 or abs(current - previous) / abs(previous) < UNCHANGED_TOLERANCE:
        return "sin cambio"
    improved = (current > previous) == key_higher_is_better
    return "mejora" if improved else "empeora"


# --------------------------------------------------------------------------- sections


def render_header(meta: RunMeta, checks_by_scenario: dict[str, list[Check]]) -> str:
    passed, total = count_passed(checks_by_scenario)
    state = "✅ APROBADO" if passed == total else "❌ HAY COMPROBACIONES FALLIDAS"
    dirty = " (con cambios sin confirmar)" if meta.dirty else ""
    return (
        "# Azuma Router — Informe de pruebas de extremo a extremo\n\n"
        f"- **Fecha:** {meta.date}\n"
        f"- **Versión:** {meta.version} · rama `{meta.branch}` · commit `{meta.commit}`{dirty}\n"
        f"- **Resultado global:** {state} — {passed} de {total} comprobaciones superadas\n\n"
        "Este informe resume qué hace hoy la herramienta cuando se la usa de verdad: se arranca "
        "el servicio real, se le envía tráfico simulado y se mide lo que ocurre. Se genera con "
        "un solo comando (ver «Cómo reproducirlo») y cada sprint añade una versión nueva, de "
        "modo que se puede seguir la evolución."
    )


def render_summary(
    checks_by_scenario: dict[str, list[Check]], results: Results, history: list[HistoryEntry]
) -> str:
    rows = [[SCENARIO_TITLES[name], verdict(group)] for name, group in checks_by_scenario.items()]
    learning = results.learning
    last = learning.blocks[-1]
    return (
        "## Resumen ejecutivo\n\n"
        "Azuma Router decide, para cada petición, a qué modelo de lenguaje enviarla, y aprende "
        "de los resultados para decidir mejor la vez siguiente, respetando las cuotas de "
        "suscripción.\n\n"
        + table(["Escenario", "Resultado"], rows)
        + "\n\n**En una frase por escenario:**\n\n"
        f"- **Aprende:** tras {learning.rounds:,} peticiones elige el mejor modelo en el "
        f"{last['optimal_rate']:.0%} de los casos y su error es "
        f"{last['mean_regret'] / learning.random_baseline_regret:.0%} del que tendría repartiendo "
        "al azar.\n"
        "- **Respeta la cuota:** frena por sí mismo el uso del modelo con cuota limitada y nunca "
        "la supera.\n"
        "- **Resiste caídas:** tras un apagado brusco no se pierde ningún dato confirmado.\n"
        "- **Resiste entradas hostiles:** mensajes malformados o abusivos se rechazan con un "
        "error claro y el servicio sigue funcionando."
    ) + _first_report_note(history)


def _first_report_note(history: list[HistoryEntry]) -> str:
    if history:
        return ""
    return "\n\n> Es el primer informe: la sección «Evolución» se irá llenando en los siguientes."


def render_world() -> str:
    quality_rows = [
        [ARM_NAMES[profile.arm_id]]
        + [f"{profile.quality_by_task[task]:.2f}" for task in world.TASK_KINDS]
        for profile in world.ARM_PROFILES
    ]
    best_rows = [
        [TASK_NAMES[task], ARM_NAMES[world.best_arm(_typical_request(task))]]
        for task in world.TASK_KINDS
    ]
    return (
        "## Qué se ha probado y cómo\n\n"
        "No se usan modelos reales: se construye un **mundo simulado** con tres modelos cuyo "
        "comportamiento verdadero conocemos (calidad, precio y velocidad por tipo de tarea). "
        "Como conocemos la respuesta correcta, podemos medir cuánto se equivoca el router. "
        "El servicio que se prueba es el **real**: se arranca como proceso independiente y se "
        "le habla por su interfaz estándar, igual que lo haría una aplicación cliente.\n\n"
        "**Calidad de cada modelo por tipo de tarea (0 a 1, más es mejor):**\n\n"
        + table(["Modelo"] + [TASK_NAMES[t] for t in world.TASK_KINDS], quality_rows)
        + "\n\n**Mejor modelo para cada tarea una vez ponderados calidad, coste y velocidad "
        "(es la respuesta que el router debe descubrir solo):**\n\n"
        + table(["Tarea", "Mejor modelo"], best_rows)
    )


def _typical_request(task: str) -> world.SimRequest:
    payload = {"task_kind": task, "input_tokens": 1500, "expected_output_tokens": 500}
    return world.SimRequest(payload, 1500, 500)


def render_learning(result: scenarios.LearningResult, group: list[Check]) -> str:
    block_rows = [
        [
            f"{int(b['start']):,}-{int(b['end']):,}",
            f"{b['mean_regret']:.4f}",
            f"{b['optimal_rate']:.1%}",
        ]
        for b in result.blocks
    ]
    task_rows = [
        [
            TASK_NAMES[task],
            ARM_NAMES[summary["oracle_arm"]],
            f"{summary['optimal_rate']:.1%}",
        ]
        for task, summary in result.per_task.items()
    ]
    return (
        f"## {SCENARIO_TITLES['learning']} — {verdict(group)}\n\n"
        f"**Qué se hace:** se envían {result.rounds:,} peticiones variadas (editar código, "
        "planificar, resumir, extraer datos y conversar). Tras cada una se informa al router "
        "de cómo le fue al modelo elegido. Sin cuotas que limiten la elección.\n\n"
        "**Qué esperamos:** que al principio se equivoque, que aprenda con la experiencia y que "
        "al final elija casi siempre el mejor modelo para cada tipo de tarea.\n\n"
        "El «error» (en inglés *regret*) es lo que se pierde por no haber elegido el mejor "
        "modelo posible en cada petición: 0 sería perfecto.\n\n"
        + table(["Peticiones", "Error medio", "Acierta el mejor modelo"], block_rows)
        + "\n\n"
        + _regret_chart(result)
        + "\n\n**Qué ha aprendido por tipo de tarea (último tramo):**\n\n"
        + table(["Tarea", "Mejor modelo real", "Aciertos"], task_rows)
        + "\n\n"
        + _speed_and_offline_evaluation(result)
        + "\n\n**Comprobaciones:**\n\n"
        + checks_table(group)
    )


def _regret_chart(result: scenarios.LearningResult) -> str:
    labels = ", ".join(f'"{int(b["start"])}-{int(b["end"])}"' for b in result.blocks)
    regrets = ", ".join(f"{b['mean_regret']:.4f}" for b in result.blocks)
    baseline = ", ".join(f"{result.random_baseline_regret:.4f}" for _ in result.blocks)
    top = max(result.random_baseline_regret, result.blocks[0]["mean_regret"]) * 1.1
    return (
        "```mermaid\nxychart-beta\n"
        '    title "Error medio por tramo de peticiones (línea inferior: router; superior: azar)"\n'
        f"    x-axis [{labels}]\n"
        f'    y-axis "Error" 0 --> {top:.2f}\n'
        f"    line [{regrets}]\n"
        f"    line [{baseline}]\n"
        "```"
    )


def _speed_and_offline_evaluation(result: scenarios.LearningResult) -> str:
    speed_rows = [
        [
            method,
            f"{result.latency[method]['p50_ms']:.2f}",
            f"{result.latency[method]['p99_ms']:.2f}",
        ]
        for method in ("select", "report")
    ]
    ope_rows = [
        [
            f"{estimate['candidate_alpha']:.1f}"
            + (" (la usada)" if estimate["candidate_alpha"] == result.alpha else ""),
            f"{estimate['ips']:.4f}",
            f"{estimate['ess']:.0f}",
        ]
        for estimate in result.ope
    ]
    return (
        "**Velocidad** (incluye la comunicación con el servicio; "
        f"~{result.throughput_rounds_per_s:,.0f} peticiones por segundo):\n\n"
        + table(["Operación", "Mediana (ms)", "Percentil 99 (ms)"], speed_rows)
        + "\n\n**Evaluación offline:** con el registro de decisiones se puede estimar, sin "
        "volver a ejecutar nada, qué recompensa habría conseguido el router con otro grado de "
        "exploración (más alto = prueba más modelos distintos). Sirve para decidir ajustes sin "
        "arriesgar tráfico real.\n\n"
        + table(["Exploración", "Recompensa media estimada", "Decisiones efectivas"], ope_rows)
    )


def render_quota(result: scenarios.QuotaResult, group: list[Check]) -> str:
    share_rows = [
        [f"{i * scenarios.QUOTA_BLOCK + 1}-{(i + 1) * scenarios.QUOTA_BLOCK}", f"{share:.0%}"]
        for i, share in enumerate(result.codex_share_by_block)
    ]
    return (
        f"## {SCENARIO_TITLES['quota']} — {verdict(group)}\n\n"
        f"**Qué se hace:** el mejor modelo para programar es una suscripción con cuota limitada "
        f"({result.capacity_units:.0f} unidades en la prueba). Se envían {result.rounds} "
        "peticiones de código y después se fuerza su uso hasta agotarla.\n\n"
        "**Qué esperamos:** que el router *racione* ese modelo a medida que la cuota se agota, "
        "enviando el resto del tráfico al siguiente mejor, y que la cuota jamás se supere.\n\n"
        "Porcentaje de peticiones de código enviadas al modelo con cuota:\n\n"
        + table(["Peticiones", "Uso del modelo con cuota"], share_rows)
        + f"\n\nCuota consumida antes de forzar: {result.free_phase_codex_pulls} de "
        f"{result.capacity_units:.0f}. Al forzar su uso se llegó a exactamente "
        f"{result.total_codex_pulls}, y la petición siguiente se rechazó con el motivo "
        f"«{result.forced_masked_reason}».\n\n**Comprobaciones:**\n\n" + checks_table(group)
    )


def render_recovery(result: scenarios.RecoveryResult, group: list[Check]) -> str:
    rows = [
        [
            case.name,
            str(case.acknowledged_updates),
            str(case.state_version_after_restart),
            f"{case.pending_after_restart} de {case.pending_at_crash}",
            "snapshot" if case.restored_from_snapshot else "registro completo",
        ]
        for case in result.cases
    ]
    return (
        f"## {SCENARIO_TITLES['recovery']} — {verdict(group)}\n\n"
        "**Qué se hace:** se mata el servicio de golpe (sin permitirle guardar nada), con "
        "algunas decisiones aún sin resultado, y se vuelve a arrancar.\n\n"
        "**Qué esperamos:** que no se pierda nada de lo ya confirmado, que las decisiones "
        "pendientes sigan pudiéndose completar y que funcione tanto si había una copia de "
        "seguridad (*snapshot*) como si no.\n\n"
        + table(
            [
                "Caso",
                "Aprendizajes confirmados",
                "Recuperados",
                "Decisiones pendientes",
                "Recuperado desde",
            ],
            rows,
        )
        + "\n\n**Comprobaciones:**\n\n"
        + checks_table(group)
    )


def render_hostile(result: scenarios.HostileResult, group: list[Check]) -> str:
    return (
        f"## {SCENARIO_TITLES['hostile']} — {verdict(group)}\n\n"
        f"**Qué se hace:** se envían {len(result.probes)} mensajes malformados, inválidos o "
        "abusivos (JSON roto, valores imposibles, mensajes de más de 1 MiB…).\n\n"
        "**Qué esperamos:** un error claro para cada uno, que el servicio no se caiga y que "
        "ningún mensaje malo estropee el siguiente. Incluye los dos fallos de seguridad "
        "corregidos en este proyecto (un mensaje demasiado largo ya no «se come» al "
        "siguiente, y los errores internos ya no revelan detalles al cliente).\n\n"
        "**Comprobaciones:**\n\n" + checks_table(group)
    )


def render_evolution(
    meta: RunMeta, results: Results, history: list[HistoryEntry], total: tuple[int, int]
) -> str:
    current = HistoryEntry(
        meta.date, meta.commit, total[0], total[1], key_metrics(results.learning)
    )
    entries = [*history, current]
    rows = [
        [e.date, f"`{e.commit}`", f"{e.checks_passed}/{e.checks_total}"]
        + [fmt.format(e.metrics[key]) for key, _, fmt, _ in METRICS]
        for e in entries
    ]
    header = ["Fecha", "Commit", "Comprobaciones"] + [label for _, label, _, _ in METRICS]
    return "## Evolución\n\n" + table(header, rows) + "\n\n" + _delta_section(history, current)


def _delta_section(history: list[HistoryEntry], current: HistoryEntry) -> str:
    if not history:
        return "_Aún no hay un informe anterior con el que comparar._"
    previous = history[-1]
    rows = [
        [
            label,
            fmt.format(previous.metrics[key]),
            fmt.format(current.metrics[key]),
            change_label(higher, previous.metrics[key], current.metrics[key]),
        ]
        for key, label, fmt, higher in METRICS
    ]
    return f"**Cambios respecto al informe del {previous.date}:**\n\n" + table(
        ["Indicador", "Antes", "Ahora", "Valoración"], rows
    )


LIMITATIONS = """## Alcance y limitaciones

- **El mundo es simulado.** No se han usado modelos de lenguaje reales ni tráfico de clientes: \
las cifras demuestran que la herramienta *funciona como está diseñada*, no cuánto ahorrará en \
producción. Una prueba con modelos reales es el siguiente paso natural.
- **Un solo proceso.** Se prueba un servicio con un único cliente; no se ha probado carga \
concurrente de varios clientes.
- **Pendientes de la auditoría de seguridad del servicio** (decisiones aplazadas, no olvidadas):
  - Límite al número de decisiones pendientes (hoy puede crecer sin tope).
  - Comprobación de propietario y permisos del directorio de datos al arrancar.
  - Lectura del registro de decisiones en streaming (hoy se carga entero al recuperar).
- **Cuota:** el router *raciona* el modelo con cuota de forma suave; lo que garantiza no \
superarla es el rechazo duro al llegar al tope."""

GLOSSARY_TERMS: tuple[tuple[str, str], ...] = (
    ("Router", "El servicio que elige a qué modelo de lenguaje enviar cada petición."),
    (
        "Bandido contextual (LinUCB)",
        "Técnica que aprende por prueba y error usando las características de cada petición.",
    ),
    (
        "Error (*regret*)",
        "Lo que se pierde por no elegir el mejor modelo en una petición. 0 es perfecto.",
    ),
    ("Recompensa", "Puntuación de un resultado: premia la calidad y penaliza coste y lentitud."),
    (
        "Cuota / precio sombra",
        "Límite de uso de una suscripción. El «precio sombra» es un coste ficticio que sube "
        "a medida que se agota, para repartir su uso.",
    ),
    ("Snapshot", "Copia de seguridad del estado aprendido."),
    (
        "Registro de decisiones (WAL)",
        "Diario donde se anota cada decisión y resultado antes de aplicarlo; permite "
        "recuperar el estado tras una caída.",
    ),
    (
        "Evaluación offline (OPE)",
        "Estimar el rendimiento de otra configuración con el diario de decisiones, sin "
        "ejecutar tráfico nuevo.",
    ),
    ("Percentil 99 (p99)", "El 99 % de las operaciones fue más rápido que este tiempo."),
)
GLOSSARY = "## Glosario\n\n" + table(
    ["Término", "Qué significa"], [list(t) for t in GLOSSARY_TERMS]
)

REPRODUCE = """## Cómo reproducirlo

```bash
uv sync --extra dev
uv run pytest -m e2e                  # ejecuta y valida los escenarios (≈10 s)
uv run python scripts/e2e_report.py   # regenera este informe y guarda sus resultados
```

Cada ejecución escribe `docs/reports/e2e/<fecha>-e2e-report.md` y `<fecha>-results.json`; \
los JSON anteriores alimentan la sección «Evolución». Las semillas son fijas: los resultados \
de aprendizaje son reproducibles; los tiempos varían según la máquina."""


# --------------------------------------------------------------------------- assembly & CLI


def render_report(meta: RunMeta, results: Results, history: list[HistoryEntry]) -> str:
    by_scenario = all_checks(results)
    sections = [
        render_header(meta, by_scenario),
        render_summary(by_scenario, results, history),
        render_world(),
        render_learning(results.learning, by_scenario["learning"]),
        render_quota(results.quota, by_scenario["quota"]),
        render_recovery(results.recovery, by_scenario["recovery"]),
        render_hostile(results.hostile, by_scenario["hostile"]),
        render_evolution(meta, results, history, count_passed(by_scenario)),
        LIMITATIONS,
        GLOSSARY,
        REPRODUCE,
    ]
    return "\n\n".join(sections) + "\n"


def write_report(output_dir: Path, meta: RunMeta, results: Results) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(output_dir, meta.date)
    path = report_path(output_dir, meta.date)
    path.write_text(render_report(meta, results, history), encoding="utf-8")
    results_path(output_dir, meta.date).write_text(
        json.dumps(results_json(meta, results), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the e2e sprint-delivery report.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="azuma-e2e-") as work_dir:
        results = run_all(Path(work_dir))
    path = write_report(args.output_dir, collect_meta(args.date), results)
    passed, total = count_passed(all_checks(results))
    print(f"{path} — {passed}/{total} comprobaciones superadas")
    return 0 if passed == total else 1
