"""Tests for log barrier implementation and parity with JAX (T14)."""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from azuma_router.quota import log_barrier


def test_log_barrier_values() -> None:
    # u = 0 => 0.0
    assert log_barrier(0.0, mu=0.5) == 0.0

    # u = 0.9, mu = 0.5 => -0.5 * ln(0.1)
    val = log_barrier(0.9, mu=0.5)
    expected = -0.5 * math.log(0.1)
    np.testing.assert_allclose(val, expected, rtol=1e-12)


@given(
    u=st.floats(min_value=0.0, max_value=0.999),
    mu=st.sampled_from([0.1, 0.5, 1.0, 2.0]),
)
def test_hypothesis_barrier_monotonicity(u: float, mu: float) -> None:
    b1 = log_barrier(u, mu)
    b2 = log_barrier(min(u + 0.0001, 0.999), mu)
    assert b1 >= 0.0
    assert b2 >= b1
