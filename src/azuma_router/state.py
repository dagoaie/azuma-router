"""Bandit state initialization.

Complies with contract §3.3, §4.2, FR-07, and ADR-0001.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from azuma_router.config import Config, to_static_config
from azuma_router.types import F32, F64, I32, U32, BanditState, QuotaView, StaticConfig


def init_state(
    config: Config | StaticConfig,
    num_arms: int | None = None,
    seed: int | None = None,
) -> BanditState:
    """Initializes a new BanditState with ridge regularization prior.

    Args:
        config: Loaded Config or StaticConfig instance.
        num_arms: Number of active arms (0 < num_arms <= k_max). If Config is passed,
            defaults to len(config.arms).
        seed: Optional seed override; if None, uses config.bandit.seed (or static seed).

    Returns:
        BanditState PyTree with exact shape and dtype invariants.
    """
    if isinstance(config, Config):
        static = to_static_config(config)
        k_arms = len(config.arms) if num_arms is None else num_arms
        s_val = config.bandit.seed if seed is None else seed
    else:
        static = config
        k_arms = static.k_max if num_arms is None else num_arms
        s_val = 42 if seed is None else seed

    k_max = static.k_max
    d = static.d
    ridge = static.ridge_lambda

    # A[a] = ridge_lambda * I_d
    eye = jnp.eye(d, dtype=F64)
    a_init = jnp.tile(eye, (k_max, 1, 1)) * ridge

    # b = zeros(k_max, d)
    b = jnp.zeros((k_max, d), dtype=F64)

    # n_pulls = zeros(k_max)
    n_pulls = jnp.zeros(k_max, dtype=I32)

    # mask: True for active arms, False for padding
    mask_list = [True] * k_arms + [False] * (k_max - k_arms)
    mask = jnp.array(mask_list, dtype=bool)

    # rng_key
    rng_key = jax.random.PRNGKey(s_val)

    return BanditState(A=a_init, b=b, n_pulls=n_pulls, mask=mask, rng_key=rng_key)


def state_from_numpy(tensors: dict[str, np.ndarray]) -> BanditState:
    """Reconstructs a BanditState PyTree from NumPy arrays (D18 boundary)."""
    return BanditState(
        A=jnp.array(tensors["A"], dtype=F64),
        b=jnp.array(tensors["b"], dtype=F64),
        n_pulls=jnp.array(tensors["n_pulls"], dtype=I32),
        mask=jnp.array(tensors["mask"], dtype=bool),
        rng_key=jnp.array(tensors["rng_key"], dtype=U32),
    )


def state_to_numpy(state: BanditState) -> dict[str, np.ndarray]:
    """Converts a BanditState PyTree to a dict of NumPy arrays."""
    return {
        "A": np.array(state.A, dtype=np.float64),
        "b": np.array(state.b, dtype=np.float64),
        "n_pulls": np.array(state.n_pulls, dtype=np.int32),
        "mask": np.array(state.mask, dtype=bool),
        "rng_key": np.array(state.rng_key, dtype=np.uint32),
    }


def make_quota_view(
    utilization: np.ndarray | list[float] | list[list[float]],
    est_units: np.ndarray | list[float],
    allowed: np.ndarray | list[bool],
    k_max: int = 32,
    w_max: int = 4,
) -> QuotaView:
    """Constructs a QuotaView PyTree from NumPy inputs (D18 boundary)."""
    u_np = np.array(utilization, dtype=np.float32)
    if u_np.ndim == 1:
        u_np = u_np.reshape((k_max, w_max))
    return QuotaView(
        utilization=jnp.array(u_np, dtype=F32),
        est_units=jnp.array(est_units, dtype=F32),
        allowed=jnp.array(allowed, dtype=bool),
    )


def make_warmup_inputs(static: StaticConfig) -> tuple[jnp.ndarray, QuotaView]:
    """Constructs dummy x array and QuotaView for JIT warm-up (D18 boundary)."""
    dummy_x = jnp.zeros(32, dtype=F64)
    dummy_qv = QuotaView(
        utilization=jnp.zeros((static.k_max, static.w_max), dtype=F32),
        est_units=jnp.zeros(static.k_max, dtype=F32),
        allowed=jnp.ones(static.k_max, dtype=bool),
    )
    return dummy_x, dummy_qv


def to_jax_array(arr: np.ndarray | list[float], dtype: Any = F64) -> jnp.ndarray:
    """Converts a NumPy array or list of floats to a JAX array (D18 boundary)."""
    return jnp.array(arr, dtype=dtype)
