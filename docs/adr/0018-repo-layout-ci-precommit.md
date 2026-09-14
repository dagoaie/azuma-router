# ADR-0018 — Layout `src/`, `pyproject` con extras `dev`, CI matriz 3.12/3.13 ubuntu+macos, pre-commit ruff+mypy, x64 en `__init__`

**Estado:** Aceptado (Grill R2 · Q18) · 2026-09-14

## Decisión
- `src/azuma_router/{__init__,config,types,featurizer,state,linucb,quota,reward,wal,snapshot,ope,daemon}.py`; `tests/` espejo; `tests/bench/`; `fixtures/`.
- `pyproject.toml`: `requires-python = ">=3.12"`, deps `jax`, `safetensors`; extra `dev`: `pytest`, `hypothesis`, `pytest-benchmark`, `ruff`, `mypy`. `uv.lock` versionado; CI usa `uv sync --frozen`.
- CI GitHub Actions: matriz `{3.12, 3.13} × {ubuntu-latest, macos-latest}`; jobs `lint`, `typecheck`, `test`, `bench` (bench solo informa, no falla en CI).
- `jax.config.update("jax_enable_x64", True)` en `azuma_router/__init__.py`; test que importa el paquete y verifica `jnp.zeros(1).dtype == float64`.
- `.pre-commit-config.yaml` con `ruff` (lint+format) y `mypy --strict` desde el primer commit.
- Rama de trabajo del slice S1: `feat/linucb-core` (creada por `convoy spin` desde `main`, que incluye el commit de Fase 0 y el bootstrap del toolchain).

## Porqué
Layout `src/` evita importar el paquete sin instalar; x64 en `__init__` garantiza que ningún módulo cree arrays float32 por accidente antes de la activación.

## Rechazado / diferido
- `jax_enable_x64` por contexto (`jax.experimental.enable_x64`): rechazado (dual behavior).
