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
  - `errors.py`: Domain exception hierarchy (`AzumaError`, `ConfigError`, `InvalidContextError`, `IncompatibleSnapshotError`, `UnsafeStateError`, `RpcError`)
  - `domain.py`: JAX-free domain enums (`TaskKind`, `Priority`) and frozen dataclasses (`RoutingRequest`, `StaticConfig`, `Pending`, `SnapshotMeta`) — MUST NOT import jax; this is the OPE import boundary (FR-34, JAX-1)
  - `types.py`: re-exports `domain.py` for the JAX path's convenience, plus NamedTuples (`BanditState`, `QuotaView`, `Diagnostics`) and float64/int32/uint32 dtypes; imports `jax`
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
- `samples/`: review-pipeline experiment with deliberately seeded bugs; excluded from Ruff

## Specs & Decisions

- `openspec/specs/<capability>/spec.md`: the 9 current capability specs (featurizer, journal, LinUCB, OPE, quota, reward, config, daemon, snapshot)
- `openspec/changes/archive/2026-09-15-linucb-core/`: `proposal.md`, `design.md` (D1–D18, follow-ups F1–F14), `tasks.md`
- `docs/contracts/linucb-core.md`: normative contract (FR/AC numbering)
- `docs/adr/0001…0018`: architecture decisions, indexed in `docs/adr/README.md`
- `CONTEXT.md`: domain glossary; all names in code, tests and docs must match it
- `proposal/tier-a/`: executed change plan and proposal

## Code Style & Guardrails

- Hard-cut policy: single canonical codepath, zero compatibility shims for historical state.
- Pure JAX math inside JIT (`linucb.py`), pure Python/NumPy outside JIT (`quota.py`, `wal.py`, `snapshot.py`, `ope.py`).
- Float64 globally active (`jax_enable_x64`).
- All function parameters and return values must have explicit type annotations.
- Python 3.12 syntax and formatting enforced by Ruff and Mypy strict mode.
