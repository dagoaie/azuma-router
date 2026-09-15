"""Tests for reward scalarization (T15 & AC-16)."""

from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from azuma_router.config import load_config, to_static_config
from azuma_router.reward import compute_reward


def test_reward_reference_value_ac16() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)

    # quality = 0.9, cost_usd = 0.0, latency_s = 11.2
    # w_q = 1.0, w_c = 0.3, w_l = 0.2, latency_budget = 30.0
    r = compute_reward(quality=0.9, cost_usd=0.0, latency_s=11.2, static=static)

    # 1.0*0.9 - 0.3*0 - 0.2*(11.2/30.0) = 0.9 - 0.07466666666666667 = 0.8253333333333334
    expected = 0.8253333333333334
    np.testing.assert_allclose(r, expected, rtol=1e-9, atol=1e-9)


@given(
    quality=st.floats(min_value=0.0, max_value=1.0),
    cost_usd=st.floats(min_value=0.0, max_value=10.0),
    latency_s=st.floats(min_value=0.0, max_value=300.0),
)
def test_hypothesis_reward_range(quality: float, cost_usd: float, latency_s: float) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    static = to_static_config(cfg)

    r = compute_reward(quality=quality, cost_usd=cost_usd, latency_s=latency_s, static=static)
    assert 0.0 <= r <= 1.0
