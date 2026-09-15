"""Tests for JIT invariants and single-compilation guarantee (T10 & AC-12)."""

from pathlib import Path

import jax

from azuma_router.config import load_config, to_static_config
from azuma_router.linucb import _compile_count, select, select_jit, update, update_jit
from azuma_router.state import init_state, make_quota_view, to_jax_array


def test_eager_calls_do_not_increment_compile_count() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    initial_select_count = _compile_count["select"]
    initial_update_count = _compile_count["update"]

    current_state = state
    for i in range(1000):
        x = [0.0] * 32
        x[i % 32] = float(i % 10) / 10.0
        x_jax = to_jax_array(x)
        qv = make_quota_view([0.0] * (32 * 4), [0.0] * 32, [True] * 32)

        current_state, arm, _prop, _diag = select(current_state, x_jax, qv, static)
        current_state = update(current_state, x_jax, arm, 0.8, static)

    assert _compile_count["select"] == initial_select_count
    assert _compile_count["update"] == initial_update_count


def test_single_compilation_guarantee_ac12() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)
    state = init_state(cfg)

    # Earlier tests may already have compiled for this static config; drop every
    # cached executable so this test observes the compilations it triggers itself.
    jax.clear_caches()
    start_select_count = _compile_count["select"]
    start_update_count = _compile_count["update"]

    current_state = state
    for i in range(10000):
        x = [0.0] * 32
        x[i % 32] = float(i % 10) / 10.0
        x_jax = to_jax_array(x)
        qv = make_quota_view([0.0] * (32 * 4), [0.0] * 32, [True] * 32)

        current_state, arm, _prop, _diag = select_jit(current_state, x_jax, qv, static)
        current_state = update_jit(current_state, x_jax, arm, 0.8, static)

    assert _compile_count["select"] - start_select_count == 1
    assert _compile_count["update"] - start_update_count == 1
