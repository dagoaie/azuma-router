"""Tests for LinUCB select logic, scoring, and sampling (T07 & T08)."""

import dataclasses
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import select, select_jit
from azuma_router.state import init_state
from azuma_router.types import F32, F64, QuotaView


def test_bonus_and_shadow_ac07_ac08() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    x = np.zeros(32, dtype=np.float64)
    x[0] = 1.0
    x[1] = 0.5
    x[16] = 1.0
    norm_x = float(np.linalg.norm(x))

    util = np.zeros((32, 4), dtype=np.float32)
    est_units = np.zeros(32, dtype=np.float32)
    allowed = np.ones(32, dtype=bool)

    qv = QuotaView(
        utilization=jnp.array(util, dtype=F32),
        est_units=jnp.array(est_units, dtype=F32),
        allowed=jnp.array(allowed, dtype=bool),
    )

    _state_prime, _arm, _propensity, diag = select(state, jnp.array(x, dtype=F64), qv, static)

    # AC-07: bonus == ||x||_2 for active arms
    assert math.isclose(float(diag.bonus[0]), norm_x, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(float(diag.bonus[1]), norm_x, rel_tol=1e-12, abs_tol=1e-12)
    assert float(diag.shadow_price[0]) == 0.0
    assert float(diag.shadow_price[1]) == 0.0

    # AC-08: u = 0.9, mu = 0.5 for arm 0
    util[0, 0] = 0.9
    qv90 = QuotaView(
        utilization=jnp.array(util, dtype=F32),
        est_units=jnp.array(est_units, dtype=F32),
        allowed=jnp.array(allowed, dtype=bool),
    )
    _, _, _, diag90 = select(state, jnp.array(x, dtype=F64), qv90, static)
    expected_shadow = -0.5 * np.log(1.0 - float(util[0, 0]))
    np.testing.assert_allclose(float(diag90.shadow_price[0]), expected_shadow, rtol=1e-9, atol=1e-9)


def test_pureness_ac09() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    x = jnp.zeros(32, dtype=F64).at[0].set(1.0)
    qv = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.ones(32, dtype=bool),
    )

    state1, arm1, prop1, diag1 = select(state, x, qv, static)
    _state2, arm2, prop2, diag2 = select(state, x, qv, static)

    assert arm1 == arm2
    assert prop1 == prop2
    np.testing.assert_array_equal(diag1.score, diag2.score)
    np.testing.assert_array_equal(diag1.policy, diag2.policy)
    # rng_key advances and is non-identical to initial
    assert not np.array_equal(np.array(state1.rng_key), np.array(state.rng_key))


def test_no_arm_available_ac10() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    x = jnp.zeros(32, dtype=F64).at[0].set(1.0)
    qv_forbidden = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.zeros(32, dtype=bool),  # all False
    )

    _, arm, prop, diag = select(state, x, qv_forbidden, static)
    assert bool(diag.no_arm_available) is True
    assert prop == 0.0
    assert not np.isnan(arm)


def test_sampling_frequency() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    x = jnp.zeros(32, dtype=F64).at[0].set(1.0)
    qv = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.ones(32, dtype=bool),
    )

    _, _, _, diag = select(state, x, qv, static)
    expected_policy = np.array(diag.policy)

    current_state = state
    pulls = [0, 0]
    n_samples = 20_000

    for _ in range(n_samples):
        current_state, arm, _, _ = select_jit(current_state, x, qv, static)
        pulls[int(arm)] += 1

    freqs = np.array(pulls) / n_samples
    np.testing.assert_allclose(freqs, expected_policy[:2], atol=0.02)


def test_softmax_policy_stays_finite_for_any_positive_temperature() -> None:
    """R4-SF-1: dividing by tau before subtracting the max overflows to inf - inf = NaN."""
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    # score / tau overflows to inf here; a subnormal tau is not used because XLA
    # flushes float64 subnormals to zero (and MIN_TAU keeps it unreachable anyway).
    static = dataclasses.replace(to_static_config(cfg), tau=1e-300)
    state = init_state(cfg)
    x = np.zeros(32, dtype=np.float64)
    x[0] = 1e9
    qv = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.ones(32, dtype=bool),
    )

    _state, _arm, propensity, diag = select(state, jnp.array(x, dtype=F64), qv, static)

    assert np.isfinite(np.asarray(diag.policy)).all()
    assert math.isclose(float(jnp.sum(diag.policy)), 1.0, rel_tol=1e-9)
    assert math.isfinite(float(propensity)) and float(propensity) > 0.0
