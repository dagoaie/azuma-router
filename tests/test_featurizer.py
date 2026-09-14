"""Tests for featurizer v1 (T04 & T05)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from azuma_router.errors import InvalidContextError
from azuma_router.featurizer import FEATURIZER_VERSION, featurize
from azuma_router.types import Priority, RoutingRequest, TaskKind


def test_featurizer_version() -> None:
    assert FEATURIZER_VERSION == 1


def test_v1_reference_vector() -> None:
    fixture_path = Path("fixtures/featurizer/v1_reference.json")
    with fixture_path.open() as f:
        data = json.load(f)

    req_data = data["request"]
    expected_x = np.array(data["expected_x"], dtype=np.float64)

    req = RoutingRequest(
        task_kind=TaskKind(req_data["task_kind"]),
        input_tokens=req_data["input_tokens"],
        expected_output_tokens=req_data["expected_output_tokens"],
        has_code=req_data["has_code"],
        has_tools=req_data["has_tools"],
        has_images=req_data["has_images"],
        agent_depth=req_data["agent_depth"],
        diff_lines=req_data["diff_lines"],
        priority=Priority(req_data["priority"]),
    )

    x = featurize(req)
    assert x.shape == (32,)
    assert x.dtype == np.float64
    np.testing.assert_allclose(x, expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_equal(x[24:32], 0.0)
    assert math.isclose(x[23], x[16] * x[1], abs_tol=1e-12)


def test_optional_fields_none() -> None:
    req = RoutingRequest(
        task_kind=TaskKind.CHAT,
        input_tokens=100,
        expected_output_tokens=None,
        has_code=False,
        has_tools=False,
        has_images=False,
        agent_depth=0,
        diff_lines=None,
        priority=Priority.LOW,
    )
    x = featurize(req)
    assert x[2] == 0.0
    assert x[3] == 0.0


def test_input_tokens_saturation() -> None:
    req = RoutingRequest(
        task_kind=TaskKind.CHAT,
        input_tokens=10_000_000,
        expected_output_tokens=None,
        has_code=False,
        has_tools=False,
        has_images=False,
        agent_depth=0,
        diff_lines=None,
        priority=Priority.LOW,
    )
    x = featurize(req)
    assert x[1] == 1.0


def test_fail_fast_invalid_context_ac04() -> None:
    with pytest.raises(InvalidContextError) as exc_info:
        RoutingRequest(
            task_kind="unknown",  # type: ignore[arg-type]
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


@given(
    task_kind=st.sampled_from(list(TaskKind)),
    input_tokens=st.integers(min_value=0, max_value=10_000_000),
    expected_output_tokens=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000)),
    has_code=st.booleans(),
    has_tools=st.booleans(),
    has_images=st.booleans(),
    agent_depth=st.integers(min_value=0, max_value=100),
    diff_lines=st.one_of(st.none(), st.integers(min_value=0, max_value=100_000)),
    priority=st.sampled_from(list(Priority)),
)
def test_hypothesis_range_properties(
    task_kind: TaskKind,
    input_tokens: int,
    expected_output_tokens: int | None,
    has_code: bool,
    has_tools: bool,
    has_images: bool,
    agent_depth: int,
    diff_lines: int | None,
    priority: Priority,
) -> None:
    req = RoutingRequest(
        task_kind=task_kind,
        input_tokens=input_tokens,
        expected_output_tokens=expected_output_tokens,
        has_code=has_code,
        has_tools=has_tools,
        has_images=has_images,
        agent_depth=agent_depth,
        diff_lines=diff_lines,
        priority=priority,
    )
    x = featurize(req)
    assert x.shape == (32,)
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert np.linalg.norm(x) <= math.sqrt(32)
