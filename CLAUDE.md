# CLAUDE.md — azuma-router

Contextual Bandit LLM Router compiled in JAX with quota-aware shadow pricing.

## Commands

- Install dependencies: `uv sync --extra dev`
- Run unit and integration tests: `uv run pytest`
- Run latency benchmarks: `uv run pytest -m bench`
- Run linter: `uv run ruff check`
- Run type checker: `uv run mypy src`
- Run daemon: `uv run python -m azuma_router.daemon --config azuma.toml`

## Project Structure

- `src/azuma_router/`: Package source
  - `errors.py`: Domain exception hierarchy (`AzumaError`, `ConfigError`, `InvalidContextError`, `IncompatibleSnapshotError`, `RpcError`)
  - `types.py`: Domain enums (`TaskKind`, `Priority`), frozen dataclasses (`RoutingRequest`, `StaticConfig`, `Pending`, `SnapshotMeta`), NamedTuples (`BanditState`, `QuotaView`, `Diagnostics`), and float64/int32/uint32 dtypes
  - `config.py`: Strict TOML parser (`load_config`) and static config projection (`to_static_config`)
  - `featurizer.py`: Context featurizer mapping `RoutingRequest` -> float64[32] vector with layout v1 (`FEATURIZER_VERSION = 1`)
  - `state.py`: Bandit initial state constructor with ridge regularization prior (`init_state`)
  - `linucb.py`: Vectorized JAX LinUCB solver, logarithmic shadow barrier, softmax policy sampling, rank-1 ridge update (`select_jit`, `update_jit`)
  - `quota.py`: Consumption ledger, rolling & fixed reset windows, logarithmic barrier (`QuotaLedger`, `log_barrier`)
  - `reward.py`: Q-C-L scalarized reward computation (`compute_reward`)
  - `wal.py`: JSONL write-ahead decision log with rotation and WAL replay (`DecisionLog`, `PendingStore`, `replay_log`)
  - `snapshot.py`: Atomic safetensors + JSON snapshot persistence and fail-fast restore (`write_snapshot`, `restore_snapshot`)
  - `daemon.py`: JSON-RPC 2.0 single-thread stdio daemon (`Router`, `serve`, `main`)
  - `ope.py`: Pure NumPy Off-Policy Evaluation IPS and Doubly-Robust estimators (`evaluate`)
- `fixtures/`: Config, featurizer reference vector, WAL replay, and OPE test fixtures
- `tests/`: Pytest test suite mirroring `src/` layout

## Code Style & Guardrails

- Hard-cut policy: single canonical codepath, zero compatibility shims for historical state.
- Pure JAX math inside JIT (`linucb.py`), pure Python/NumPy outside JIT (`quota.py`, `wal.py`, `snapshot.py`, `ope.py`).
- Float64 globally active (`jax_enable_x64`).
- All function parameters and return values must have explicit type annotations.
- Python 3.12 syntax and formatting enforced by Ruff and Mypy strict mode.
