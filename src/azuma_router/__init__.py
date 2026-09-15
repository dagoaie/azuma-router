"""azuma-router: contextual-bandit LLM router compiled in JAX with quota-aware shadow pricing.

Attribute access below is lazy (PEP 562): merely `import azuma_router` or one of its
submodules — e.g. `azuma_router.ope`, which MUST NOT pull in JAX (FR-34, JAX-1) — does not
import JAX. `jax_enable_x64` (ADR-0002/0018) is instead enabled inline by every module that
actually touches `jax.numpy` (`types.py`, `linucb.py`, `state.py`), immediately before that
module's own `import jax.numpy`, so the switch lands regardless of which of those three loads
first. Bare `import azuma_router` alone does not build any array and so does not need to flip
the switch itself; it lands the first time a jax-backed export (e.g. `select`) is touched.
"""

from typing import Any

__all__ = [
    "BanditState",
    "QuotaLedger",
    "StaticConfig",
    "restore",
    "select",
    "snapshot",
    "update",
]

_LAZY_ATTRS = {
    "select": "azuma_router.linucb",
    "update": "azuma_router.linucb",
    "QuotaLedger": "azuma_router.quota",
    "restore": "azuma_router.snapshot",
    "snapshot": "azuma_router.snapshot",
    "BanditState": "azuma_router.types",
    "StaticConfig": "azuma_router.domain",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_ATTRS[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None

    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)
