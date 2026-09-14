"""Tests for domain types (T01 & T02)."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given
from hypothesis import strategies as st

from azuma_router.errors import InvalidContextError
from azuma_router.types import (
    F64,
    I32,
    U32,
    BanditState,
    Priority,
    RoutingRequest,
    SnapshotMeta,
    StaticConfig,
    TaskKind,
)


def test_task_kind_enum_order() -> None:
    expected = [
        "code_edit",
        "code_review",
        "code_gen",
        "test_gen",
        "planning",
        "summarize",
        "qa",
        "extract",
        "classify",
        "agentic_tool_loop",
        "chat",
        "other",
    ]
    assert [k.value for k in TaskKind] == expected
    assert len(TaskKind) == 12


def test_priority_enum() -> None:
    assert [p.value for p in Priority] == ["low", "normal", "high"]


def test_routing_request_valid() -> None:
    req = RoutingRequest(
        task_kind=TaskKind.CODE_EDIT,
        input_tokens=100,
        expected_output_tokens=50,
        has_code=True,
        has_tools=False,
        has_images=False,
        agent_depth=2,
        diff_lines=10,
        priority=Priority.NORMAL,
    )
    assert req.task_kind == TaskKind.CODE_EDIT
    assert req.expected_output_tokens == 50


def test_routing_request_optionals_none() -> None:
    req = RoutingRequest(
        task_kind=TaskKind.CHAT,
        input_tokens=0,
        expected_output_tokens=None,
        has_code=False,
        has_tools=False,
        has_images=False,
        agent_depth=0,
        diff_lines=None,
        priority=Priority.LOW,
    )
    assert req.expected_output_tokens is None
    assert req.diff_lines is None


def test_routing_request_negative_tokens() -> None:
    with pytest.raises(InvalidContextError) as exc_info:
        RoutingRequest(
            task_kind=TaskKind.CHAT,
            input_tokens=-1,
            expected_output_tokens=None,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=0,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc_info.value.field == "input_tokens"


def test_routing_request_negative_expected_output_tokens() -> None:
    with pytest.raises(InvalidContextError) as exc_info:
        RoutingRequest(
            task_kind=TaskKind.CHAT,
            input_tokens=10,
            expected_output_tokens=-5,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=0,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc_info.value.field == "expected_output_tokens"


def test_routing_request_negative_agent_depth() -> None:
    with pytest.raises(InvalidContextError) as exc_info:
        RoutingRequest(
            task_kind=TaskKind.CHAT,
            input_tokens=10,
            expected_output_tokens=None,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=-1,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc_info.value.field == "agent_depth"


def test_routing_request_invalid_field_types() -> None:
    # Boolean input_tokens
    with pytest.raises(InvalidContextError) as exc1:
        RoutingRequest(
            task_kind=TaskKind.CHAT,
            input_tokens=True,  # type: ignore[arg-type]
            expected_output_tokens=None,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=0,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc1.value.field == "input_tokens"

    # String input_tokens
    with pytest.raises(InvalidContextError) as exc2:
        RoutingRequest(
            task_kind=TaskKind.CHAT,
            input_tokens="100",  # type: ignore[arg-type]
            expected_output_tokens=None,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=0,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc2.value.field == "input_tokens"


def test_routing_request_string_task_kind() -> None:
    # If passed a string matching a TaskKind, coerce or raise if invalid
    req = RoutingRequest(
        task_kind="code_edit",  # type: ignore[arg-type]
        input_tokens=10,
        expected_output_tokens=None,
        has_code=False,
        has_tools=False,
        has_images=False,
        agent_depth=0,
        diff_lines=None,
        priority="normal",  # type: ignore[arg-type]
    )
    assert req.task_kind == TaskKind.CODE_EDIT
    assert req.priority == Priority.NORMAL

    with pytest.raises(InvalidContextError) as exc_info:
        RoutingRequest(
            task_kind="invalid_kind",  # type: ignore[arg-type]
            input_tokens=10,
            expected_output_tokens=None,
            has_code=False,
            has_tools=False,
            has_images=False,
            agent_depth=0,
            diff_lines=None,
            priority=Priority.LOW,
        )
    assert exc_info.value.field == "task_kind"


def test_static_config_frozen_and_hashable() -> None:
    sc = StaticConfig(
        k_max=32,
        w_max=4,
        d=32,
        ridge_lambda=1.0,
        alpha=1.0,
        tau=0.05,
        epsilon_floor=0.02,
        w_quality=1.0,
        w_cost=0.3,
        w_latency=0.2,
        cost_budget_usd=0.5,
        latency_budget_s=30.0,
        barrier_mu=(0.5,) + (0.0,) * 31,
        is_subscription=(True,) + (False,) * 31,
    )
    assert hash(sc) is not None
    with pytest.raises(AttributeError):
        sc.k_max = 64  # type: ignore[misc]


def test_bandit_state_pytree() -> None:
    state = BanditState(
        A=jnp.zeros((32, 32, 32), dtype=F64),
        b=jnp.zeros((32, 32), dtype=F64),
        n_pulls=jnp.zeros(32, dtype=I32),
        mask=jnp.ones(32, dtype=bool),
        rng_key=jnp.array([0, 42], dtype=U32),
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 5
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, BanditState)


def test_snapshot_meta_roundtrip() -> None:
    sc = StaticConfig(
        k_max=32,
        w_max=4,
        d=32,
        ridge_lambda=1.0,
        alpha=1.0,
        tau=0.05,
        epsilon_floor=0.02,
        w_quality=1.0,
        w_cost=0.3,
        w_latency=0.2,
        cost_budget_usd=0.5,
        latency_budget_s=30.0,
        barrier_mu=(0.5,) + (0.0,) * 31,
        is_subscription=(True,) + (False,) * 31,
    )
    meta = SnapshotMeta(
        schema_version=1,
        state_version=42,
        featurizer_version=1,
        arm_ids=["gpt-5-codex@openai-codex"],
        static=sc,
        seed=42,
        created_at="2026-09-14T12:00:00Z",
        ledger={"gpt-5-codex@openai-codex": []},
        config_sha256="abc123hash",
    )
    d = meta.to_dict()
    restored = SnapshotMeta.from_dict(d)
    assert restored == meta


@given(
    input_tokens=st.integers(),
    expected_output_tokens=st.one_of(st.none(), st.integers()),
    agent_depth=st.integers(),
    diff_lines=st.one_of(st.none(), st.integers()),
)
def test_routing_request_integer_bounds_property(
    input_tokens: int,
    expected_output_tokens: int | None,
    agent_depth: int,
    diff_lines: int | None,
) -> None:
    values = [input_tokens, agent_depth, expected_output_tokens, diff_lines]
    in_bounds = all(v is None or 0 <= v <= 2**31 - 1 for v in values)

    def build() -> RoutingRequest:
        return RoutingRequest(
            task_kind=TaskKind.CODE_EDIT,
            input_tokens=input_tokens,
            expected_output_tokens=expected_output_tokens,
            has_code=True,
            has_tools=False,
            has_images=False,
            agent_depth=agent_depth,
            diff_lines=diff_lines,
            priority=Priority.NORMAL,
        )

    if in_bounds:
        assert build().input_tokens == input_tokens
    else:
        with pytest.raises(InvalidContextError):
            build()
