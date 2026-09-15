"""Tests for off-policy evaluation IPS and DR estimators (T26 & AC-28)."""

import ast
import json
from dataclasses import replace
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


def test_ope_preserves_each_arm_logged_quota_penalty(tmp_path: Path) -> None:
    """Candidate scores retain the per-arm quota penalty already encoded in `score`."""
    static = replace(
        to_static_config(load_config(Path("fixtures/config/azuma.valid.toml"))), tau=1.0
    )
    log_path = tmp_path / "heterogeneous-quota.jsonl"
    entries = [
        {
            "phase": "select",
            "decision_id": "decision-1",
            "arm_index": 0,
            "propensity": 0.5,
            "alpha": 1.0,
            "policy": [0.5, 0.5],
            "score": [10.0, 0.0],
            "exploit": [0.0, 0.0],
            "bonus": [0.0, 0.0],
            "radius": [0.0, 0.0],
            "active": [True, True],
            "shadow_price": [0.0, 10.0],
            "est_units": 0.0,
        },
        {"phase": "report", "decision_id": "decision-1", "reward": 1.0},
    ]
    log_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    result = evaluate(log_path, static)

    assert result["ips"] > 1.9
    assert result["dr"] > 1.9


def test_ope_reconstructs_the_exploration_bonus_of_an_alpha_zero_log_from_its_radius(
    tmp_path: Path,
) -> None:
    """OPE-1: `alpha=0` logs a zero bonus for every arm; only the persisted, unscaled confidence
    radius lets a positive-alpha candidate rebuild its own exploration term (FR-34)."""
    static = replace(
        to_static_config(load_config(Path("fixtures/config/azuma.valid.toml"))),
        alpha=2.0,
        tau=1.0,
        epsilon_floor=0.0,
    )
    log_path = tmp_path / "alpha-zero.jsonl"
    entries = [
        {
            "phase": "select",
            "decision_id": "decision-1",
            "arm_index": 0,
            "propensity": 0.5,
            "alpha": 0.0,
            "policy": [0.5, 0.5],
            "score": [0.0, 0.0],
            "exploit": [0.0, 0.0],
            "bonus": [0.0, 0.0],
            "radius": [0.0, 5.0],
            "active": [True, True],
            "shadow_price": [0.0, 0.0],
            "est_units": 0.0,
        },
        {"phase": "report", "decision_id": "decision-1", "reward": 1.0},
    ]
    log_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    result = evaluate(log_path, static)

    # By hand: bonus_cand = 2.0 * [0, 5] = [0, 10]; score_cand = [0, 10]; softmax(tau=1,
    # eps=0) over both active arms gives pi_cand[0] = 1 / (1 + e^10) ~= 4.54e-5;
    # weight = pi_cand[0] / mu_t(0.5) ~= 9.08e-5; ips = weight * reward(1.0).
    # A candidate that (like the pre-fix code) leaves bonus_cand at the logged 0 for every arm
    # would instead reproduce the uniform logged policy exactly and report ips == 1.0.
    assert result["ips"] < 1e-3
    assert result["dr"] < 1e-3


def test_ope_uses_the_persisted_active_mask_not_the_logged_policy_to_find_active_arms(
    tmp_path: Path,
) -> None:
    """OPE-2: an active arm whose score is far below the -1e9 inactive sentinel, or whose logged
    policy underflowed to 0 under a zero epsilon_floor, must still get its epsilon-floor share."""
    static = replace(
        to_static_config(load_config(Path("fixtures/config/azuma.valid.toml"))),
        alpha=1.0,
        tau=1.0,
        epsilon_floor=0.05,
    )
    log_path = tmp_path / "active-mask.jsonl"
    entries = [
        {
            "phase": "select",
            "decision_id": "decision-1",
            "arm_index": 0,
            "propensity": 0.01,
            "alpha": 1.0,
            "policy": [0.0, 1.0],  # arm 0 underflowed to 0 under the logging policy
            "score": [-2e9, 10.0],  # a real, structurally active but heavily quota-penalized arm
            "exploit": [0.0, 0.0],
            "bonus": [0.0, 0.0],
            "radius": [0.0, 0.0],
            "active": [True, True],  # neither exhausted nor disabled: both structurally active
            "shadow_price": [0.0, 0.0],
            "est_units": 0.0,
        },
        {"phase": "report", "decision_id": "decision-1", "reward": 1.0},
    ]
    log_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    result = evaluate(log_path, static)

    # By hand: exp((-2e9 - 10)/1) underflows to 0.0, so p_softmax = [0, 1]; the epsilon floor is
    # still split over both active arms (k_act=2): pi_cand[0] = 0.95*0 + 0.05/2 = 0.025.
    # weight = pi_cand[0] / mu_t(0.01) = 2.5; ips = weight * reward(1.0) = 2.5.
    # Inferring activity from the logged policy (arm 0's policy is 0) would instead treat arm 0 as
    # inactive, give it no floor share, and report ips == 0.0.
    np.testing.assert_allclose(result["ips"], 2.5, rtol=1e-9)
    assert result["ips"] == result["dr"]  # exploit is 0 for every arm, so dr collapses to ips


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
