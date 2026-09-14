"""T00 — package seam: importing ``azuma_router`` fixes the numeric contract (ADR-0018)."""

import jax.numpy as jnp

import azuma_router
from azuma_router import errors

PUBLIC_API = {
    "select",
    "update",
    "BanditState",
    "StaticConfig",
    "QuotaLedger",
    "snapshot",
    "restore",
}


def test_importing_package_enables_float64() -> None:
    # ADR-0002/0018: x64 must be on before any submodule builds an array.
    assert jnp.zeros(1).dtype == jnp.float64


def test_public_api_is_exactly_the_adr_0012_surface() -> None:
    assert set(azuma_router.__all__) == PUBLIC_API


def test_every_domain_error_derives_from_azuma_error() -> None:
    for name in ("ConfigError", "InvalidContextError", "IncompatibleSnapshotError", "RpcError"):
        assert issubclass(getattr(errors, name), errors.AzumaError), name
