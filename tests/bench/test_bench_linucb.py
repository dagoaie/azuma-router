"""Latency benchmarks for select_jit and update_jit (NFR-01)."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import select_jit, update_jit
from azuma_router.state import init_state
from azuma_router.types import F32, F64, QuotaView


@pytest.mark.bench
def test_bench_select_jit(benchmark: Any) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)
    x = jnp.zeros(32, dtype=F64).at[0].set(1.0)
    qv = QuotaView(
        utilization=jnp.zeros((32, 4), dtype=F32),
        est_units=jnp.zeros(32, dtype=F32),
        allowed=jnp.ones(32, dtype=bool),
    )

    # Warmup JIT
    state, _, _, _ = select_jit(state, x, qv, static)

    def run() -> None:
        select_jit(state, x, qv, static)

    benchmark(run)


@pytest.mark.bench
def test_bench_update_jit(benchmark: Any) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)
    x = jnp.zeros(32, dtype=F64).at[0].set(1.0)

    # Warmup JIT
    state = update_jit(state, x, 0, 0.8, static)

    def run() -> None:
        update_jit(state, x, 0, 0.8, static)

    benchmark(run)
