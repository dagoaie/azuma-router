"""Sublinear regret simulation test (T11)."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import select_jit, update_jit
from azuma_router.state import init_state
from azuma_router.types import F32, F64, QuotaView, StaticConfig


def test_sublinear_regret_simulation() -> None:
    fixture_path = Path("fixtures/sim/arms_synthetic.json")
    with fixture_path.open() as f:
        data = json.load(f)

    theta_stars = np.array(data["theta_stars"], dtype=np.float64)  # shape (2, 32)

    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    base_static = to_static_config(cfg)
    static = StaticConfig(
        k_max=base_static.k_max,
        w_max=base_static.w_max,
        d=base_static.d,
        ridge_lambda=base_static.ridge_lambda,
        alpha=0.2,
        tau=0.01,
        epsilon_floor=0.001,
        w_quality=base_static.w_quality,
        w_cost=base_static.w_cost,
        w_latency=base_static.w_latency,
        cost_budget_usd=base_static.cost_budget_usd,
        latency_budget_s=base_static.latency_budget_s,
        barrier_mu=base_static.barrier_mu,
        is_subscription=base_static.is_subscription,
    )
    state = init_state(cfg, seed=42)

    rng = np.random.default_rng(42)
    n_rounds = 20_000

    regrets = np.zeros(n_rounds, dtype=np.float64)
    arm_pulls = np.zeros(2, dtype=np.int32)

    qv = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.ones(32, dtype=bool),
    )

    current_state = state
    for t in range(n_rounds):
        # Generate random context x_t in [0, 1]^32 with bias x_0 = 1.0
        x = rng.uniform(0.0, 1.0, size=32)
        x[0] = 1.0
        x_jax = jnp.array(x, dtype=F64)

        # Expected rewards
        expected_rewards = theta_stars @ x
        opt_arm = int(np.argmax(expected_rewards))
        opt_reward = float(expected_rewards[opt_arm])

        # Bandit selects arm
        current_state, arm_idx, _, _ = select_jit(current_state, x_jax, qv, static)
        a = int(arm_idx)
        arm_pulls[a] += 1

        # Observed reward (with small noise)
        r_observed = float(np.clip(expected_rewards[a] + rng.normal(0, 0.02), 0.0, 1.0))
        regrets[t] = opt_reward - expected_rewards[a]

        # Bandit updates
        current_state = update_jit(current_state, x_jax, a, r_observed, static)

    cum_regret = np.cumsum(regrets)

    # Log-log slope over second half [n_rounds//2 : n_rounds]
    half = n_rounds // 2
    ts = np.arange(half, n_rounds) + 1
    log_t = np.log(ts)
    log_r = np.log(np.maximum(cum_regret[half:], 1e-9))

    poly = np.polyfit(log_t, log_r, deg=1)
    slope = poly[0]

    # Exponent of log-log fit must be < 0.6
    assert slope < 0.6, f"Cumulative regret log-log slope is {slope}, expected < 0.6"

    # Each active arm receives >= 1% of total pulls
    pull_ratio = arm_pulls / n_rounds
    assert pull_ratio[0] >= 0.01
    assert pull_ratio[1] >= 0.01
