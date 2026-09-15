"""Off-Policy Evaluation (IPS and Doubly-Robust estimators) in pure NumPy.

Complies with contract §3.9, FR-34, ADR-0008.
MUST NOT import JAX.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from azuma_router.types import StaticConfig


def evaluate(
    log_path: str | Path,
    candidate_static: StaticConfig,
) -> dict[str, Any]:
    """Evaluates candidate policy on logged decisions dataset using IPS and DR (FR-34).

    Args:
        log_path: Path to decisions.jsonl WAL file.
        candidate_static: Candidate hyperparameters to evaluate.

    Returns:
        dict containing 'ips', 'dr', 'n', and 'ess'.
    """
    path_obj = Path(log_path)
    if not path_obj.exists():
        return {"ips": 0.0, "dr": 0.0, "n": 0, "ess": 0.0}

    selects: dict[str, dict[str, Any]] = {}
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []

    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            try:
                entry = json.loads(line_str)
            except Exception:
                continue

            phase = entry.get("phase")
            d_id = entry.get("decision_id")

            if phase == "select" and d_id:
                selects[d_id] = entry
            elif phase == "report" and d_id:
                if d_id not in selects:
                    msg = f"Orphan report line without select: decision_id={d_id}"
                    raise ValueError(msg)
                paired.append((selects[d_id], entry))

    if not paired:
        return {"ips": 0.0, "dr": 0.0, "n": 0, "ess": 0.0}

    weights: list[float] = []
    ips_terms: list[float] = []
    dr_terms: list[float] = []

    for s_entry, r_entry in paired:
        exploit = np.array(s_entry["exploit"], dtype=np.float64)
        logged_bonus = np.array(s_entry["bonus"], dtype=np.float64)
        logged_score = np.array(s_entry["score"], dtype=np.float64)

        # Reconstruct candidate bonus
        logged_alpha = float(s_entry.get("alpha", 1.0))
        if logged_alpha > 0.0:
            bonus_cand = logged_bonus * (candidate_static.alpha / logged_alpha)
        else:
            bonus_cand = logged_bonus

        # The logged score already includes each arm's quota penalty. Candidate evaluation changes
        # only the exploration term, so it must not use the selected arm's units for every arm.
        score_cand = logged_score + bonus_cand - logged_bonus

        # Active arms from logged policy
        logged_policy = np.array(s_entry["policy"], dtype=np.float64)
        active = logged_policy > 0.0
        k_act = int(np.sum(active))

        if k_act == 0:
            continue

        logits = score_cand / candidate_static.tau
        logits_safe = np.where(active, logits, -1e9)
        max_logit = np.max(logits_safe)
        exp_logits = np.where(active, np.exp(logits - max_logit), 0.0)
        sum_exp = np.sum(exp_logits)
        p_softmax = np.where(sum_exp > 0.0, exp_logits / sum_exp, 0.0)

        p_floor = np.where(active, candidate_static.epsilon_floor / max(k_act, 1), 0.0)
        eps = candidate_static.epsilon_floor
        pi_cand = np.where(active, (1.0 - eps) * p_softmax + p_floor, 0.0)

        a_t = int(s_entry["arm_index"])
        mu_t = float(s_entry["propensity"])
        r_t = float(r_entry["reward"])

        pi_t = float(pi_cand[a_t])
        weight_t = pi_t / mu_t if mu_t > 0.0 else 0.0

        r_hat_t = float(exploit[a_t])
        r_hat_cand = float(np.sum(pi_cand * exploit))

        ips_val = weight_t * r_t
        dr_val = r_hat_cand + weight_t * (r_t - r_hat_t)

        weights.append(weight_t)
        ips_terms.append(ips_val)
        dr_terms.append(dr_val)

    n = len(ips_terms)
    if n == 0:
        return {"ips": 0.0, "dr": 0.0, "n": 0, "ess": 0.0}

    ips_mean = float(np.mean(ips_terms))
    dr_mean = float(np.mean(dr_terms))

    w_arr = np.array(weights)
    sum_w = np.sum(w_arr)
    sum_w2 = np.sum(w_arr**2)
    ess = float((sum_w**2) / sum_w2) if sum_w2 > 0.0 else 0.0

    return {
        "ips": ips_mean,
        "dr": dr_mean,
        "n": n,
        "ess": ess,
    }
