"""Deterministically transforms a RoutingRequest to context vector x_t in [0, 1]^32.

Complies with contract §3.2, §4.2, ADR-0004, ADR-0013, ADR-0015.
"""

import math

import numpy as np

from azuma_router.domain import Priority, RoutingRequest, TaskKind
from azuma_router.errors import InvalidContextError

FEATURIZER_VERSION = 1
D = 32

_TASK_KIND_ORDER: tuple[TaskKind, ...] = (
    TaskKind.CODE_EDIT,
    TaskKind.CODE_REVIEW,
    TaskKind.CODE_GEN,
    TaskKind.TEST_GEN,
    TaskKind.PLANNING,
    TaskKind.SUMMARIZE,
    TaskKind.QA,
    TaskKind.EXTRACT,
    TaskKind.CLASSIFY,
    TaskKind.AGENTIC_TOOL_LOOP,
    TaskKind.CHAT,
    TaskKind.OTHER,
)

_PRIORITY_ORDER: tuple[Priority, ...] = (
    Priority.LOW,
    Priority.NORMAL,
    Priority.HIGH,
)


def featurize(request: RoutingRequest) -> np.ndarray:
    """Transforms a RoutingRequest into a float64[32] context vector.

    Raises:
        InvalidContextError: If any field is invalid or out-of-domain.
    """
    if not isinstance(request, RoutingRequest):
        msg = f"Expected RoutingRequest, got {type(request).__name__}"
        raise InvalidContextError("request", msg)

    x = np.zeros(D, dtype=np.float64)

    # 0: bias
    x[0] = 1.0

    # 1: input_tokens: min(log10(1+n)/6, 1)
    if request.input_tokens < 0:
        msg = f"input_tokens must be >= 0, got {request.input_tokens}"
        raise InvalidContextError("input_tokens", msg)
    x[1] = min(math.log10(1.0 + float(request.input_tokens)) / 6.0, 1.0)

    # 2: expected_output_tokens: min(log10(1+n)/6, 1) or 0.0 if None
    if request.expected_output_tokens is not None:
        if request.expected_output_tokens < 0:
            msg = f"expected_output_tokens must be >= 0, got {request.expected_output_tokens}"
            raise InvalidContextError("expected_output_tokens", msg)
        x[2] = min(math.log10(1.0 + float(request.expected_output_tokens)) / 6.0, 1.0)
    else:
        x[2] = 0.0

    # 3: diff_lines: min(log10(1+n)/4, 1) or 0.0 if None
    if request.diff_lines is not None:
        if request.diff_lines < 0:
            msg = f"diff_lines must be >= 0, got {request.diff_lines}"
            raise InvalidContextError("diff_lines", msg)
        x[3] = min(math.log10(1.0 + float(request.diff_lines)) / 4.0, 1.0)
    else:
        x[3] = 0.0

    # 4-15: task_kind one-hot
    try:
        tk_idx = _TASK_KIND_ORDER.index(request.task_kind)
    except ValueError as exc:
        msg = f"Unknown task_kind: {request.task_kind!r}"
        raise InvalidContextError("task_kind", msg) from exc
    x[4 + tk_idx] = 1.0

    # 16-18: has_code, has_tools, has_images
    x[16] = 1.0 if request.has_code else 0.0
    x[17] = 1.0 if request.has_tools else 0.0
    x[18] = 1.0 if request.has_images else 0.0

    # 19: agent_depth: min(n, 8)/8
    if request.agent_depth < 0:
        msg = f"agent_depth must be >= 0, got {request.agent_depth}"
        raise InvalidContextError("agent_depth", msg)
    x[19] = min(float(request.agent_depth), 8.0) / 8.0

    # 20-22: priority one-hot
    try:
        p_idx = _PRIORITY_ORDER.index(request.priority)
    except ValueError as exc:
        msg = f"Unknown priority: {request.priority!r}"
        raise InvalidContextError("priority", msg) from exc
    x[20 + p_idx] = 1.0

    # 23: interaction: x[16] * x[1]
    x[23] = x[16] * x[1]

    # 24-31: padding (already 0.0)

    # Validate output range and finiteness (FR-06)
    if not np.all(np.isfinite(x)):
        msg = "Featurizer produced non-finite values"
        raise InvalidContextError("context_vector", msg)

    if np.any(x < 0.0) or np.any(x > 1.0):
        msg = "Featurizer produced values outside [0, 1]"
        raise InvalidContextError("context_vector", msg)

    return x
