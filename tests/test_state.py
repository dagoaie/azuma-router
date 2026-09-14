"""Tests for initial state construction (T06 & AC-06)."""

from pathlib import Path

import numpy as np

from azuma_router.config import load_config
from azuma_router.state import init_state
from azuma_router.types import F64, I32, U32, BanditState


def test_init_state_ac06() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    state = init_state(cfg)

    assert isinstance(state, BanditState)
    assert state.A.shape == (32, 32, 32)
    assert state.A.dtype == F64
    assert state.b.shape == (32, 32)
    assert state.b.dtype == F64
    assert state.n_pulls.shape == (32,)
    assert state.n_pulls.dtype == I32
    assert state.mask.shape == (32,)
    assert state.mask.dtype == bool
    assert state.rng_key.shape == (2,)
    assert state.rng_key.dtype == U32

    # A[a] = ridge_lambda * I for all a
    ridge = cfg.bandit.ridge_lambda
    eye = ridge * np.eye(32, dtype=np.float64)
    for a in range(32):
        np.testing.assert_array_equal(np.array(state.A[a]), eye)

    # b == 0, n_pulls == 0
    np.testing.assert_array_equal(np.array(state.b), 0.0)
    np.testing.assert_array_equal(np.array(state.n_pulls), 0)

    # mask[:2] == True (2 active arms), mask[2:] == False (padding)
    assert bool(state.mask[0]) is True
    assert bool(state.mask[1]) is True
    assert not np.any(np.array(state.mask[2:]))
