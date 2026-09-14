"""Tests for JIT invariants and single-compilation guarantee (T10 & AC-12)."""

from pathlib import Path

import jax.numpy as jnp

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import _compile_count, select_jit, update_jit
from azuma_router.state import init_state
from azuma_router.types import F32, F64, QuotaView


def test_single_compilation_guarantee_ac12() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    reset_count = _compile_count.copy()

    current_state = state
    for i in range(100):
        x = jnp.zeros(32, dtype=F64).at[i % 32].set(float(i % 10) / 10.0)
        qv = QuotaView(
            utilization=jnp.zeros((32, 4), dtype=F32),
            est_units=jnp.zeros(32, dtype=F32),
            allowed=jnp.ones(32, dtype=bool),
        )

        current_state, arm, _prop, _diag = select_jit(current_state, x, qv, static)
        current_state = update_jit(current_state, x, arm, 0.8, static)

    assert _compile_count["select"] - reset_count.get("select", 0) == 1
    assert _compile_count["update"] - reset_count.get("update", 0) == 1
