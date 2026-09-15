"""Scalarized Quality-Cost-Latency reward computation.

Complies with contract §3.5, FR-18, ADR-0007.
"""

import numpy as np

from azuma_router.types import StaticConfig


def compute_reward(
    quality: float,
    cost_usd: float,
    latency_s: float,
    static: StaticConfig,
) -> float:
    """Computes composite reward scalar in [0, 1] from quality, cost, and latency.

    Args:
        quality: Caller-provided quality score in [0, 1].
        cost_usd: Observed monetary cost in USD (0.0 for subscription arms).
        latency_s: Observed end-to-end latency in seconds.
        static: Static hyperparameters containing weights and budgets.

    Returns:
        Scalarized reward in [0, 1].
    """
    q_term = static.w_quality * quality

    cost_ratio = min(cost_usd / static.cost_budget_usd, 1.0) if static.cost_budget_usd > 0 else 0.0
    c_term = static.w_cost * cost_ratio

    if static.latency_budget_s > 0:
        lat_ratio = min(latency_s / static.latency_budget_s, 1.0)
    else:
        lat_ratio = 0.0
    l_term = static.w_latency * lat_ratio

    raw_reward = q_term - c_term - l_term
    return float(np.clip(raw_reward, 0.0, 1.0))
