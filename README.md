# azuma-router

High-performance Contextual Bandit LLM Router compiled in JAX with quota-aware shadow pricing.

Azuma routes LLM calls across multiple models and providers in < 1 ms by scalarizing Quality-Cost-Latency (Q-C-L) rewards, estimating UCB confidence bounds in JAX float64, and dynamically penalizing subscription models with logarithmic barrier shadow prices to prevent quota exhaustion.

---

## 1. Quickstart

### Installation

Requires Python >= 3.12 and `uv`:

```bash
uv sync --extra dev
```

### Configuration (`azuma.toml`)

Create an `azuma.toml` configuration file:

```toml
schema_version = 1

[bandit]
seed = 42
k_max = 32
w_max = 4
d = 32
ridge_lambda = 1.0
alpha = 1.0
tau = 0.05
epsilon_floor = 0.02
w_quality = 1.0
w_cost = 0.3
w_latency = 0.2
cost_budget_usd = 0.50
latency_budget_s = 30.0
pending_ttl_s = 3600
snapshot_dir = "./.azuma"
snapshot_every_updates = 100
snapshot_every_s = 30
decision_log_path = "./.azuma/decisions.jsonl"
decision_log_max_bytes = 268435456

[[arms]]
model_id = "gpt-5-codex"
provider = "openai-codex"
billing_mode = "subscription"
price_in_per_1m = 0.0
price_out_per_1m = 0.0
units_per_request = 1.0
units_per_1k_tokens = 0.0
barrier_mu = 0.5

  [[arms.windows]]
  kind = "rolling"
  duration_s = 18000
  capacity = 300.0

  [[arms.windows]]
  kind = "fixed_reset"
  period = "weekly"
  reset_anchor = "2026-09-08T00:00:00Z"
  capacity = 1500.0

[[arms]]
model_id = "qwen3-coder"
provider = "openrouter"
billing_mode = "per_token"
price_in_per_1m = 0.30
price_out_per_1m = 1.20
units_per_request = 0.0
units_per_1k_tokens = 0.0
barrier_mu = 0.0
```

### Running the JSON-RPC Daemon

Launch the daemon over stdio:

```bash
uv run python -m azuma_router.daemon --config azuma.toml
```

Upon boot, the daemon performs WAL replay and JIT warm-up, emitting a `ready` notification on `stdout`:

```json
{"jsonrpc":"2.0","method":"ready","params":{"state_version":0,"arms":["gpt-5-codex@openai-codex","qwen3-coder@openrouter"],"restored_from_snapshot":false,"pending_replayed":0,"warmup_ms":79.5}}
```

---

## 2. Architecture & Design

1. **JAX JIT Boundary & Shape Invariance**: Linear bandit calculations (`select_jit`, `update_jit`) are vectorised with `jax.vmap` over static shape `K_max = 32, d = 32`. A Cholesky solver in float64 (`cho_factor` / `cho_solve`) maintains numerical stability.
2. **Quota Shadow Pricing**: A logarithmic convex barrier $\phi(u) = -\mu_a \ln(1 - u)$ dynamically penalizes subscription arms as rolling/fixed quota utilization $u \to 1$.
3. **WAL & Atomic Snapshots**: Every selection is logged to `decisions.jsonl` before responding. State is persisted in `state.safetensors` and `state.json` via atomic directory renaming.
4. **Off-Policy Evaluation (OPE)**: Evaluates candidate policies offline using Inverse Propensity Scoring (IPS) and Doubly Robust (DR) estimators without importing JAX.

---

## 3. Protocol (JSON-RPC 2.0 over stdio)

- **`select`**: Request model selection for a given task context.
- **`report`**: Report observed quality, cost, and latency for a decision ID.
- **`snapshot`**: Force an immediate atomic snapshot write.
- **`health`**: Query current state version, pending count, arm utilizations, and JIT compilation count.

---

## 4. Development & Testing

```bash
# Run test suite
uv run pytest

# Run latency benchmarks (NFR-01)
uv run pytest -m bench

# Lint and typecheck
uv run ruff check
uv run mypy src
```
