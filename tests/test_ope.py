"""Tests for off-policy evaluation IPS and DR estimators (T26 & AC-28)."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config, to_static_config
from azuma_router.ope import evaluate
from azuma_router.types import StaticConfig


def test_ope_same_policy_ac28() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    base_static = to_static_config(cfg)
    logged_static = StaticConfig(
        k_max=base_static.k_max,
        w_max=base_static.w_max,
        d=base_static.d,
        ridge_lambda=base_static.ridge_lambda,
        alpha=0.2,
        tau=0.1,
        epsilon_floor=0.02,
        w_quality=base_static.w_quality,
        w_cost=base_static.w_cost,
        w_latency=base_static.w_latency,
        cost_budget_usd=base_static.cost_budget_usd,
        latency_budget_s=base_static.latency_budget_s,
        barrier_mu=base_static.barrier_mu,
        is_subscription=base_static.is_subscription,
    )
    log_path = Path("fixtures/ope/decisions.jsonl")

    res = evaluate(log_path, logged_static)

    # Read mean logged reward from reports
    rewards = []
    with log_path.open() as f:
        for line in f:
            data = json.loads(line)
            if data.get("phase") == "report":
                rewards.append(data["reward"])

    mean_reward = float(np.mean(rewards))

    assert res["n"] == 500
    np.testing.assert_allclose(res["ips"], mean_reward, rtol=0.02, atol=0.02)
    np.testing.assert_allclose(res["dr"], mean_reward, rtol=0.02, atol=0.02)
    assert res["ess"] == 500.0


def test_ope_different_policy() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    base_static = to_static_config(cfg)
    log_path = Path("fixtures/ope/decisions.jsonl")

    logged_static = StaticConfig(
        k_max=base_static.k_max,
        w_max=base_static.w_max,
        d=base_static.d,
        ridge_lambda=base_static.ridge_lambda,
        alpha=0.2,
        tau=0.1,
        epsilon_floor=0.02,
        w_quality=base_static.w_quality,
        w_cost=base_static.w_cost,
        w_latency=base_static.w_latency,
        cost_budget_usd=base_static.cost_budget_usd,
        latency_budget_s=base_static.latency_budget_s,
        barrier_mu=base_static.barrier_mu,
        is_subscription=base_static.is_subscription,
    )

    # Candidate with different alpha
    cand_static = StaticConfig(
        k_max=base_static.k_max,
        w_max=base_static.w_max,
        d=base_static.d,
        ridge_lambda=base_static.ridge_lambda,
        alpha=2.0,
        tau=0.1,
        epsilon_floor=0.02,
        w_quality=base_static.w_quality,
        w_cost=base_static.w_cost,
        w_latency=base_static.w_latency,
        cost_budget_usd=base_static.cost_budget_usd,
        latency_budget_s=base_static.latency_budget_s,
        barrier_mu=base_static.barrier_mu,
        is_subscription=base_static.is_subscription,
    )

    res_logged = evaluate(log_path, logged_static)
    res_cand = evaluate(log_path, cand_static)

    assert res_cand["n"] == 500
    assert res_cand["ips"] != res_logged["ips"]
    assert res_cand["dr"] != res_logged["dr"]
    assert res_cand["ess"] < res_cand["n"]


def test_ope_orphan_report(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)

    log_path = tmp_path / "orphan.jsonl"
    orphan_line = {
        "phase": "report",
        "decision_id": "orphan-uuid",
        "ts": 100.0,
        "state_version": 1,
        "reward": 0.5,
    }
    log_path.write_text(json.dumps(orphan_line) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        evaluate(log_path, static)
    assert "orphan-uuid" in str(exc_info.value)


def test_ope_no_jax_import() -> None:
    ope_file = Path("src/azuma_router/ope.py")
    tree = ast.parse(ope_file.read_text(encoding="utf-8"))

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    assert not any("jax" in mod for mod in imports), f"ope.py imports jax: {imports}"
