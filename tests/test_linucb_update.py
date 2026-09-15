"""Tests for LinUCB rank-1 update (T09 & AC-11)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import update
from azuma_router.state import init_state
from azuma_router.types import F64


def test_single_arm_update_isolation() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    x = np.zeros(32, dtype=np.float64)
    x[0] = 1.0
    x[1] = 0.5
    reward = 0.8
    arm = 0

    state_next = update(state, jnp.array(x, dtype=F64), arm, reward, static)

    # Arm 0 modified
    assert int(state_next.n_pulls[0]) == 1
    np.testing.assert_array_equal(np.array(state_next.b[0]), x * reward)

    # Arm 1 unmodified
    assert int(state_next.n_pulls[1]) == 0
    np.testing.assert_array_equal(np.array(state_next.b[1]), 0.0)
    np.testing.assert_array_equal(np.array(state_next.A[1]), np.array(state.A[1]))


@given(
    arm=st.integers(min_value=0, max_value=1),
    reward=st.floats(min_value=0.0, max_value=1.0),
)
def test_hypothesis_1000_updates_ac11(arm: int, reward: float) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    rng = np.random.default_rng(42)

    current_state = state
    for _ in range(50):  # Run 50 random updates per test case iteration
        x_val = rng.uniform(0.0, 1.0, size=32)
        current_state = update(current_state, jnp.array(x_val, dtype=F64), arm, reward, static)

    for a in range(2):
        a_mat = np.array(current_state.A[a])
        b_a = np.array(current_state.b[a])

        # Bitwise symmetry
        np.testing.assert_array_equal(a_mat, a_mat.T)

        # Cholesky factor does not fail
        _l_factor = np.linalg.cholesky(a_mat)

        # Solve error ||A theta_hat - b||_inf < 1e-9
        theta_hat = np.linalg.solve(a_mat, b_a)
        residual = np.linalg.norm(a_mat @ theta_hat - b_a, ord=np.inf)
        assert residual < 1e-9
