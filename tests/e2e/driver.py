"""Closed-loop driver: select -> simulated model call -> report, scored against the oracle."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e2e import world
from e2e.client import DaemonProcess


@dataclass(frozen=True)
class RoundRecord:
    """One completed routing round, with the router's choice scored against the oracle."""

    task_kind: str
    chosen_arm: str
    oracle_arm: str
    oracle_reward: float
    chosen_reward: float
    observed_reward: float
    propensity: float
    decision_id: str

    @property
    def regret(self) -> float:
        return self.oracle_reward - self.chosen_reward


@contextmanager
def running_daemon(
    work_dir: Path, **config_overrides: float
) -> Iterator[tuple[DaemonProcess, dict[str, Any]]]:
    """Writes the world config under `work_dir` and boots a daemon on it.

    The daemon is always killed and reaped on exit, even if the scenario raises midway.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "azuma.toml"
    config_path.write_text(world.build_config_toml(work_dir, **config_overrides), encoding="utf-8")
    with DaemonProcess(config_path, work_dir / "daemon.stderr.log") as daemon:
        yield daemon, daemon.start()


def route_and_report(
    daemon: DaemonProcess,
    request: world.SimRequest,
    rng: np.random.Generator,
    *,
    allowed_arms: list[str] | None = None,
) -> RoundRecord:
    """Runs one full round: ask the router, simulate the call, report what happened."""
    params: dict[str, Any] = {"request": request.payload, "verbose": False}
    if allowed_arms is not None:
        params["allowed_arms"] = allowed_arms
    selection = daemon.call("select", params)
    chosen = selection["arm"]["arm_id"]
    outcome = world.observe(chosen, request, rng)
    report = daemon.call("report", report_params(selection["decision_id"], outcome))
    oracle_arm = world.best_arm(request)
    return RoundRecord(
        task_kind=request.task_kind,
        chosen_arm=chosen,
        oracle_arm=oracle_arm,
        oracle_reward=world.expected_reward(oracle_arm, request),
        chosen_reward=world.expected_reward(chosen, request),
        observed_reward=float(report["reward"]),
        propensity=float(selection["propensity"]),
        decision_id=selection["decision_id"],
    )


def report_params(decision_id: str, outcome: world.Outcome) -> dict[str, Any]:
    """Maps a simulated outcome to the `report` RPC payload."""
    return {
        "decision_id": decision_id,
        "quality": outcome.quality,
        "cost_usd": outcome.cost_usd,
        "latency_s": outcome.latency_s,
        "actual_input_tokens": outcome.input_tokens,
        "actual_output_tokens": outcome.output_tokens,
    }
