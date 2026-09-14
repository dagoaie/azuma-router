"""Bandit state initialization.

Complies with contract §3.3, §4.2, FR-07, and ADR-0001.
"""

import jax
import jax.numpy as jnp

from azuma_router.config import Config, to_static_config
from azuma_router.types import F64, I32, BanditState, StaticConfig


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
