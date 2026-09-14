"""LinUCB (Disjoint) bandit algorithm compiled in JAX.

Complies with contract §3.3, §4.2, ADR-0001, ADR-0002, ADR-0003, ADR-0008, ADR-0018.
"""

import jax
import jax.numpy as jnp

from azuma_router.types import F64, I32, BanditState, Diagnostics, QuotaView, StaticConfig

_compile_count: dict[str, int] = {"select": 0, "update": 0}


def _score_single_arm(
    a_arm: jnp.ndarray,
    b_a: jnp.ndarray,
    x: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solves ridge estimator via Cholesky decomposition for a single arm (ADR-0002)."""
    l_factor, lower = jax.scipy.linalg.cho_factor(a_arm, lower=True)
    theta_hat = jax.scipy.linalg.cho_solve((l_factor, lower), b_a)
    exploit = jnp.dot(x, theta_hat)
    z = jax.scipy.linalg.solve_triangular(l_factor, x, lower=True)
    bonus = alpha * jnp.sqrt(jnp.maximum(jnp.sum(z**2), 0.0))
    return exploit, bonus


_vmap_score_arms = jax.vmap(_score_single_arm, in_axes=(0, 0, None, None))


def _select_pure(
    state: BanditState,
    x: jnp.ndarray,
    quota_view: QuotaView,
    static: StaticConfig,
) -> tuple[BanditState, jnp.ndarray, float | jnp.ndarray, Diagnostics]:
    """Pure JAX LinUCB select step (ADR-0001, ADR-0008)."""
    _compile_count["select"] += 1

    exploit, bonus = _vmap_score_arms(state.A, state.b, x, static.alpha)

    # Shadow price via log barrier (ADR-0006)
    barrier_mu = jnp.array(static.barrier_mu, dtype=F64)
    is_sub = jnp.array(static.is_subscription, dtype=bool)

    u_clip = jnp.clip(quota_view.utilization.astype(F64), 0.0, 1.0 - 1e-6)
    mu_col = jnp.expand_dims(barrier_mu, axis=1)
    is_sub_col = jnp.expand_dims(is_sub, axis=1)

    barrier_per_w = jnp.where(is_sub_col, -mu_col * jnp.log(1.0 - u_clip), 0.0)
    shadow_price = jnp.max(barrier_per_w, axis=1)

    est_units = quota_view.est_units.astype(F64)
    raw_score = exploit + bonus - shadow_price * est_units

    # Active arm determination (FR-08, FR-15)
    util_exhausted = jnp.any(quota_view.utilization >= 1.0, axis=1)
    active = state.mask & quota_view.allowed & (~util_exhausted)
    score = jnp.where(active, raw_score, -jnp.inf)

    # Sampling policy (ADR-0008)
    k_act = jnp.sum(active.astype(I32))
    no_arm_available = k_act == 0

    logits = score / static.tau
    logits_safe = jnp.where(active, logits, -1e9)
    max_logit = jnp.max(logits_safe)
    exp_logits = jnp.where(active, jnp.exp(logits - max_logit), 0.0)
    sum_exp = jnp.sum(exp_logits)
    p_softmax = jnp.where(sum_exp > 0.0, exp_logits / sum_exp, 0.0)

    p_floor = jnp.where(active, static.epsilon_floor / jnp.maximum(k_act, 1), 0.0)
    policy = jnp.where(active, (1.0 - static.epsilon_floor) * p_softmax + p_floor, 0.0)

    rng_key, sub_key = jax.random.split(state.rng_key)
    cat_logits = jnp.where(active, jnp.log(jnp.maximum(policy, 1e-300)), -1e9)
    sampled_arm = jax.random.categorical(sub_key, cat_logits)

    arm = jnp.where(no_arm_available, 0, sampled_arm)
    propensity = jnp.where(no_arm_available, 0.0, policy[arm])
    out_policy = jnp.where(no_arm_available, jnp.zeros(static.k_max, dtype=F64), policy)

    state_prime = BanditState(
        A=state.A,
        b=state.b,
        n_pulls=state.n_pulls,
        mask=state.mask,
        rng_key=rng_key,
    )
    diagnostics = Diagnostics(
        exploit=exploit,
        bonus=bonus,
        shadow_price=shadow_price,
        score=score,
        policy=out_policy,
        no_arm_available=no_arm_available,
    )

    return state_prime, arm, propensity, diagnostics


def _update_pure(
    state: BanditState,
    x: jnp.ndarray,
    arm: int | jnp.ndarray,
    reward: float | jnp.ndarray,
    static: StaticConfig,
) -> BanditState:
    """Pure JAX LinUCB rank-1 update (FR-11, FR-12, ADR-0002)."""
    _compile_count["update"] += 1

    x_outer = jnp.outer(x, x)
    arm_idx = jnp.array(arm, dtype=I32)
    r_val = jnp.array(reward, dtype=F64)

    a_new = state.A.at[arm_idx].add(x_outer)
    b_new = state.b.at[arm_idx].add(r_val * x)
    n_pulls_new = state.n_pulls.at[arm_idx].add(1)

    return BanditState(
        A=a_new,
        b=b_new,
        n_pulls=n_pulls_new,
        mask=state.mask,
        rng_key=state.rng_key,
    )


select = _select_pure
select_jit = jax.jit(_select_pure, static_argnums=(3,))

update = _update_pure
update_jit = jax.jit(_update_pure, static_argnums=(4,))
