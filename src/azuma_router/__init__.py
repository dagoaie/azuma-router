"""azuma-router: contextual-bandit LLM router compiled in JAX with quota-aware shadow pricing.

Importing this package enables 64-bit floats globally (ADR-0002/0018). It must run before
any submodule touches ``jax.numpy``; the imports below are ordered accordingly.
"""

import jax

# jax.config.update is untyped upstream; this is the canonical x64 switch (ADR-0018).
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__all__ = [
    "BanditState",
    "QuotaLedger",
    "StaticConfig",
    "restore",
    "select",
    "snapshot",
    "update",
]
