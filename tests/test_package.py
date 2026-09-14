"""T00 — package seam: importing ``azuma_router`` fixes the numeric contract (ADR-0018)."""

import ast
from pathlib import Path

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
    for name in PUBLIC_API:
        assert hasattr(azuma_router, name)
        assert getattr(azuma_router, name) is not None


def test_every_domain_error_derives_from_azuma_error() -> None:
    for name in ("ConfigError", "InvalidContextError", "IncompatibleSnapshotError", "RpcError"):
        assert issubclass(getattr(errors, name), errors.AzumaError), name


SRC_DIR = Path("src/azuma_router")


def _imported_modules(py_file: Path) -> set[str]:
    """Top-level module names imported anywhere in the file (``import a.b`` -> ``a``)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_package_jax_import_boundary() -> None:
    allowed_jax_modules = {"__init__.py", "types.py", "state.py", "linucb.py"}

    for py_file in SRC_DIR.glob("*.py"):
        if py_file.name in allowed_jax_modules:
            continue
        jax_imports = {m for m in _imported_modules(py_file) if "jax" in m}
        assert not jax_imports, f"D18 violation: {py_file.name} imports JAX ({jax_imports})"


def test_package_prohibited_network_pickle_imports() -> None:
    banned = {"socket", "pickle", "urllib", "requests", "http", "asyncio", "netrc", "ftplib"}

    for py_file in SRC_DIR.glob("*.py"):
        violations = _imported_modules(py_file) & banned
        assert not violations, f"NFR-07 violation in {py_file.name}: {violations}"
