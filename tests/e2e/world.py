"""Simulated world: three model arms with known quality/cost/latency and a synthetic traffic mix.

The ground truth lives here, outside the router, so a scenario can compute the oracle (best
possible) reward of every request and measure the router's regret against it.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CODEX = "gpt-5-codex@openai-codex"
QWEN = "qwen3-coder@openrouter"
HAIKU = "claude-haiku@anthropic"

# Reward weights and budgets, mirrored in the generated config (contract §3.5).
W_QUALITY = 1.0
W_COST = 0.3
W_LATENCY = 0.2
COST_BUDGET_USD = 0.01
LATENCY_BUDGET_S = 30.0

QUALITY_NOISE_STD = 0.05
LATENCY_NOISE_STD = 0.10


@dataclass(frozen=True)
class ArmProfile:
    """Ground truth of one arm: expected quality per task kind, latency model and token prices."""

    arm_id: str
    quality_by_task: dict[str, float]
    latency_base_s: float
    latency_per_1k_tokens_s: float
    price_in_per_1m: float
    price_out_per_1m: float


ARM_PROFILES: tuple[ArmProfile, ...] = (
    ArmProfile(
        CODEX,
        {"code_edit": 0.92, "planning": 0.85, "summarize": 0.70, "extract": 0.70, "chat": 0.75},
        latency_base_s=10.0,
        latency_per_1k_tokens_s=2.0,
        price_in_per_1m=0.0,
        price_out_per_1m=0.0,
    ),
    ArmProfile(
        QWEN,
        {"code_edit": 0.80, "planning": 0.65, "summarize": 0.60, "extract": 0.93, "chat": 0.70},
        latency_base_s=5.0,
        latency_per_1k_tokens_s=1.5,
        price_in_per_1m=0.30,
        price_out_per_1m=1.20,
    ),
    ArmProfile(
        HAIKU,
        {"code_edit": 0.55, "planning": 0.60, "summarize": 0.90, "extract": 0.75, "chat": 0.88},
        latency_base_s=2.0,
        latency_per_1k_tokens_s=0.8,
        price_in_per_1m=0.25,
        price_out_per_1m=1.25,
    ),
)
ALL_ARM_IDS: tuple[str, ...] = tuple(profile.arm_id for profile in ARM_PROFILES)
TASK_KINDS: tuple[str, ...] = ("code_edit", "planning", "summarize", "extract", "chat")
_TASK_WEIGHTS = (0.30, 0.15, 0.20, 0.15, 0.20)
_CODE_TASKS = frozenset({"code_edit"})
_PRIORITIES = ("low", "normal", "high")


@dataclass(frozen=True)
class SimRequest:
    """One synthetic routing request plus the true token counts it will end up consuming."""

    payload: dict[str, Any]
    actual_input_tokens: int
    actual_output_tokens: int

    @property
    def task_kind(self) -> str:
        return str(self.payload["task_kind"])


@dataclass(frozen=True)
class Outcome:
    """What a model call produced: the signals the router receives in `report`."""

    quality: float
    cost_usd: float
    latency_s: float
    input_tokens: int
    output_tokens: int


def build_config_toml(
    work_dir: Path,
    *,
    codex_window_capacity: float = 300.0,
    snapshot_every_updates: int = 100,
    alpha: float = 0.3,
    seed: int = 42,
) -> str:
    """Renders the daemon config for the three-arm world; only codex is quota-limited."""
    return f"""schema_version = 1

[bandit]
seed = {seed}
k_max = 32
w_max = 4
d = 32
ridge_lambda = 1.0
alpha = {alpha}
tau = 0.05
epsilon_floor = 0.02
w_quality = {W_QUALITY}
w_cost = {W_COST}
w_latency = {W_LATENCY}
cost_budget_usd = {COST_BUDGET_USD}
latency_budget_s = {LATENCY_BUDGET_S}
pending_ttl_s = 3600
snapshot_dir = "{(work_dir / "snapshots").as_posix()}"
snapshot_every_updates = {snapshot_every_updates}
snapshot_every_s = 3600
decision_log_path = "{(work_dir / "decisions.jsonl").as_posix()}"
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
  capacity = {codex_window_capacity}

[[arms]]
model_id = "qwen3-coder"
provider = "openrouter"
billing_mode = "per_token"
price_in_per_1m = 0.30
price_out_per_1m = 1.20
units_per_request = 0.0
units_per_1k_tokens = 0.0
barrier_mu = 0.0

[[arms]]
model_id = "claude-haiku"
provider = "anthropic"
billing_mode = "per_token"
price_in_per_1m = 0.25
price_out_per_1m = 1.25
units_per_request = 0.0
units_per_1k_tokens = 0.0
barrier_mu = 0.0
"""


def make_traffic(
    rng: np.random.Generator, count: int, *, only: str | None = None
) -> list[SimRequest]:
    """Draws `count` requests; `only` pins every request to one task kind."""
    return [_draw_request(rng, only) for _ in range(count)]


def _draw_request(rng: np.random.Generator, only: str | None) -> SimRequest:
    task = only or str(rng.choice(TASK_KINDS, p=_TASK_WEIGHTS))
    input_tokens = int(np.clip(rng.lognormal(math.log(1500), 0.8), 50, 150_000))
    output_tokens = int(np.clip(rng.lognormal(math.log(500), 0.7), 10, 20_000))
    payload: dict[str, Any] = {
        "task_kind": task,
        "input_tokens": input_tokens,
        "expected_output_tokens": output_tokens,
        "has_code": task in _CODE_TASKS,
        "has_tools": bool(rng.random() < 0.2),
        "has_images": False,
        "agent_depth": int(rng.integers(0, 4)),
        "diff_lines": int(rng.integers(1, 400)) if task in _CODE_TASKS else None,
        "priority": str(rng.choice(_PRIORITIES, p=(0.2, 0.6, 0.2))),
    }
    return SimRequest(payload, input_tokens, output_tokens)


def profile_of(arm_id: str) -> ArmProfile:
    for profile in ARM_PROFILES:
        if profile.arm_id == arm_id:
            return profile
    raise KeyError(arm_id)


def _expected_signals(profile: ArmProfile, request: SimRequest) -> tuple[float, float, float]:
    quality = profile.quality_by_task[request.task_kind]
    total_tokens = request.actual_input_tokens + request.actual_output_tokens
    latency = profile.latency_base_s + profile.latency_per_1k_tokens_s * total_tokens / 1000.0
    cost = (
        profile.price_in_per_1m * request.actual_input_tokens
        + profile.price_out_per_1m * request.actual_output_tokens
    ) / 1e6
    return quality, cost, latency


def scalarize(quality: float, cost_usd: float, latency_s: float) -> float:
    """The router's Q-C-L reward (contract §3.5), reproduced to score decisions independently."""
    raw = (
        W_QUALITY * quality
        - W_COST * min(cost_usd / COST_BUDGET_USD, 1.0)
        - W_LATENCY * min(latency_s / LATENCY_BUDGET_S, 1.0)
    )
    return float(np.clip(raw, 0.0, 1.0))


def expected_reward(arm_id: str, request: SimRequest) -> float:
    """Noise-free reward an arm would earn on a request: the ground truth for regret."""
    return scalarize(*_expected_signals(profile_of(arm_id), request))


def best_arm(request: SimRequest, candidates: tuple[str, ...] = ALL_ARM_IDS) -> str:
    """The oracle arm for a request among `candidates` (all arms by default)."""
    return max(candidates, key=lambda arm_id: expected_reward(arm_id, request))


def observe(arm_id: str, request: SimRequest, rng: np.random.Generator) -> Outcome:
    """Simulates the model call: noisy quality and latency around the arm's ground truth."""
    quality, cost, latency = _expected_signals(profile_of(arm_id), request)
    noisy_quality = float(np.clip(quality + rng.normal(0.0, QUALITY_NOISE_STD), 0.0, 1.0))
    noisy_latency = max(0.1, latency * (1.0 + rng.normal(0.0, LATENCY_NOISE_STD)))
    return Outcome(
        quality=noisy_quality,
        cost_usd=cost,
        latency_s=noisy_latency,
        input_tokens=request.actual_input_tokens,
        output_tokens=request.actual_output_tokens,
    )
