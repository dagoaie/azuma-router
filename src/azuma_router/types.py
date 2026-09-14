"""Domain data types, enums, NamedTuples, and data structures.

Complies with contract §4.2 and ADR-0001, ADR-0002, ADR-0008, ADR-0009, ADR-0013, ADR-0015.
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import jax.numpy as jnp

from azuma_router.errors import InvalidContextError

# Explicit dtypes (ADR-0018)
F64 = jnp.float64
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32

MAX_INT31 = 2**31 - 1


class TaskKind(StrEnum):
    """Closed enum of 12 task kinds in exact ADR-0015 order."""

    CODE_EDIT = "code_edit"
    CODE_REVIEW = "code_review"
    CODE_GEN = "code_gen"
    TEST_GEN = "test_gen"
    PLANNING = "planning"
    SUMMARIZE = "summarize"
    QA = "qa"
    EXTRACT = "extract"
    CLASSIFY = "classify"
    AGENTIC_TOOL_LOOP = "agentic_tool_loop"
    CHAT = "chat"
    OTHER = "other"


class Priority(StrEnum):
    """Request priority enum."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class RoutingRequest:
    """Semi-structured request payload to be featurized (ADR-0013)."""

    task_kind: TaskKind
    input_tokens: int
    expected_output_tokens: int | None
    has_code: bool
    has_tools: bool
    has_images: bool
    agent_depth: int
    diff_lines: int | None
    priority: Priority

    def __post_init__(self) -> None:
        # Coerce string enum inputs if needed
        if isinstance(self.task_kind, str) and not isinstance(self.task_kind, TaskKind):
            try:
                object.__setattr__(self, "task_kind", TaskKind(self.task_kind))
            except ValueError as err:
                msg = f"Unknown task_kind: {self.task_kind!r}"
                raise InvalidContextError("task_kind", msg) from err
        elif not isinstance(self.task_kind, TaskKind):
            msg = f"Invalid task_kind type: {type(self.task_kind)}"
            raise InvalidContextError("task_kind", msg)

        if isinstance(self.priority, str) and not isinstance(self.priority, Priority):
            try:
                object.__setattr__(self, "priority", Priority(self.priority))
            except ValueError as err:
                msg = f"Unknown priority: {self.priority!r}"
                raise InvalidContextError("priority", msg) from err
        elif not isinstance(self.priority, Priority):
            msg = f"Invalid priority type: {type(self.priority)}"
            raise InvalidContextError("priority", msg)

        if not isinstance(self.has_code, bool):
            raise InvalidContextError("has_code", f"has_code must be bool, got {self.has_code!r}")

        if not isinstance(self.has_tools, bool):
            msg = f"has_tools must be bool, got {self.has_tools!r}"
            raise InvalidContextError("has_tools", msg)

        if not isinstance(self.has_images, bool):
            msg = f"has_images must be bool, got {self.has_images!r}"
            raise InvalidContextError("has_images", msg)

        if (
            not isinstance(self.input_tokens, int)
            or isinstance(self.input_tokens, bool)
            or not (0 <= self.input_tokens <= MAX_INT31)
        ):
            msg = f"input_tokens must be int between 0 and 2**31-1, got {self.input_tokens!r}"
            raise InvalidContextError("input_tokens", msg)

        if self.expected_output_tokens is not None and (
            not isinstance(self.expected_output_tokens, int)
            or isinstance(self.expected_output_tokens, bool)
            or not (0 <= self.expected_output_tokens <= MAX_INT31)
        ):
            msg = (
                "expected_output_tokens must be int between 0 and 2**31-1, "
                f"got {self.expected_output_tokens!r}"
            )
            raise InvalidContextError("expected_output_tokens", msg)

        if (
            not isinstance(self.agent_depth, int)
            or isinstance(self.agent_depth, bool)
            or not (0 <= self.agent_depth <= MAX_INT31)
        ):
            msg = f"agent_depth must be int between 0 and 2**31-1, got {self.agent_depth!r}"
            raise InvalidContextError("agent_depth", msg)

        if self.diff_lines is not None and (
            not isinstance(self.diff_lines, int)
            or isinstance(self.diff_lines, bool)
            or not (0 <= self.diff_lines <= MAX_INT31)
        ):
            msg = f"diff_lines must be int between 0 and 2**31-1, got {self.diff_lines!r}"
            raise InvalidContextError("diff_lines", msg)


@dataclass(frozen=True)
class StaticConfig:
    """Static shape parameters and hyperparameters passed to JIT via static_argnums (ADR-0001)."""

    k_max: int
    w_max: int
    d: int
    ridge_lambda: float
    alpha: float
    tau: float
    epsilon_floor: float
    w_quality: float
    w_cost: float
    w_latency: float
    cost_budget_usd: float
    latency_budget_s: float
    barrier_mu: tuple[float, ...]
    is_subscription: tuple[bool, ...]

    def __hash__(self) -> int:
        return hash(
            (
                self.k_max,
                self.w_max,
                self.d,
                self.ridge_lambda,
                self.alpha,
                self.tau,
                self.epsilon_floor,
                self.w_quality,
                self.w_cost,
                self.w_latency,
                self.cost_budget_usd,
                self.latency_budget_s,
                self.barrier_mu,
                self.is_subscription,
            )
        )


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
    shadow_price: jnp.ndarray  # f64[k_max]
    score: jnp.ndarray  # f64[k_max]
    policy: jnp.ndarray  # f64[k_max]
    no_arm_available: bool | jnp.ndarray


@dataclass(frozen=True)
class Pending:
    """In-memory representation of an un-reported decision (ADR-0016)."""

    decision_id: str
    arm_index: int
    x: list[float]
    propensity: float
    ts_select: float
    est_units: float
    ledger_event_id: int
    state_version: int
    select_entry: dict[str, Any]  # journal line, carried forward verbatim on WAL rotation


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadata saved alongside state.safetensors in state.json (ADR-0009)."""

    schema_version: int
    state_version: int
    featurizer_version: int
    arm_ids: list[str]
    static: StaticConfig
    seed: int
    created_at: str
    ledger: dict[str, Any]
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_version": self.state_version,
            "featurizer_version": self.featurizer_version,
            "arm_ids": self.arm_ids,
            "static": asdict(self.static),
            "seed": self.seed,
            "created_at": self.created_at,
            "ledger": self.ledger,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotMeta":
        static_data = data["static"]
        static = StaticConfig(
            k_max=static_data["k_max"],
            w_max=static_data["w_max"],
            d=static_data["d"],
            ridge_lambda=float(static_data["ridge_lambda"]),
            alpha=float(static_data["alpha"]),
            tau=float(static_data["tau"]),
            epsilon_floor=float(static_data["epsilon_floor"]),
            w_quality=float(static_data["w_quality"]),
            w_cost=float(static_data["w_cost"]),
            w_latency=float(static_data["w_latency"]),
            cost_budget_usd=float(static_data["cost_budget_usd"]),
            latency_budget_s=float(static_data["latency_budget_s"]),
            barrier_mu=tuple(float(x) for x in static_data["barrier_mu"]),
            is_subscription=tuple(bool(x) for x in static_data["is_subscription"]),
        )
        return cls(
            schema_version=data["schema_version"],
            state_version=data["state_version"],
            featurizer_version=data["featurizer_version"],
            arm_ids=data["arm_ids"],
            static=static,
            seed=data["seed"],
            created_at=data["created_at"],
            ledger=data["ledger"],
            config_sha256=data["config_sha256"],
        )
