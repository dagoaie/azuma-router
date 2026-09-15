"""Domain data types, enums, NamedTuples, and data structures.

Complies with contract §4.2 and ADR-0001, ADR-0002, ADR-0008, ADR-0009, ADR-0013, ADR-0015.

The JAX-free members (`TaskKind`, `Priority`, `RoutingRequest`, `MAX_INT31`, `StaticConfig`,
`Pending`, `SnapshotMeta`) live in `azuma_router.domain` and are re-exported here for the
online JAX path's convenience; import them from `domain` directly wherever JAX must not be
pulled in transitively (FR-34, JAX-1) — see `ope.py`, `wal.py`, `config.py`, `featurizer.py`.
"""

from typing import NamedTuple

import jax

# jax.config.update is untyped upstream; this is the canonical x64 switch (ADR-0018). It must
# run before any submodule touches jax.numpy -- see linucb.py/state.py for the same guard.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp  # noqa: E402

from azuma_router.domain import (  # noqa: E402
    MAX_INT31,
    Pending,
    Priority,
    RoutingRequest,
    SnapshotMeta,
    StaticConfig,
    TaskKind,
)

# Explicit dtypes (ADR-0018)
F64 = jnp.float64
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32

__all__ = [
    "F32",
    "F64",
    "I32",
    "MAX_INT31",
    "U32",
    "BanditState",
    "Diagnostics",
    "Pending",
    "Priority",
    "QuotaView",
    "RoutingRequest",
    "SnapshotMeta",
    "StaticConfig",
    "TaskKind",
]


class BanditState(NamedTuple):
    """Immutable PyTree container for dynamic bandit state arrays (ADR-0001)."""

    A: jnp.ndarray  # f64[k_max, d, d]
    b: jnp.ndarray  # f64[k_max, d]
    n_pulls: jnp.ndarray  # i32[k_max]
    mask: jnp.ndarray  # bool[k_max]
    rng_key: jnp.ndarray  # u32[2]


class QuotaView(NamedTuple):
    """Dynamic inputs to select_jit computed outside JIT by QuotaLedger (ADR-0005)."""

    utilization: jnp.ndarray  # f32[k_max, w_max]
    est_units: jnp.ndarray  # f32[k_max]
    allowed: jnp.ndarray  # bool[k_max]


class Diagnostics(NamedTuple):
    """Diagnostic breakdown produced by select (ADR-0008)."""

    exploit: jnp.ndarray  # f64[k_max]
    bonus: jnp.ndarray  # f64[k_max]
    radius: jnp.ndarray  # f64[k_max], unscaled confidence radius: bonus == alpha * radius (OPE-1)
    active: jnp.ndarray  # bool[k_max], structurally active mask (mask & allowed & !exhausted)
    shadow_price: jnp.ndarray  # f64[k_max]
    score: jnp.ndarray  # f64[k_max]
    policy: jnp.ndarray  # f64[k_max]
    no_arm_available: bool | jnp.ndarray
