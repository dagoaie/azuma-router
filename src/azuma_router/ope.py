"""Off-Policy Evaluation (IPS and Doubly-Robust estimators) in pure NumPy.

Complies with contract §3.9, FR-34, ADR-0008.
MUST NOT import JAX.
"""

from pathlib import Path
from typing import Any

import numpy as np

from azuma_router.domain import StaticConfig
from azuma_router.wal import _read_journal


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
    active_path = Path(log_path)
    # A pair fully closed before a rotation lives only in the retained rotated segment
    # (OPE-ROT-1); read it first, then the active segment, matching wal.replay_log's
    # convention (`<stem>.1<suffix>`). A select carried forward verbatim across the rotation
    # boundary is keyed by decision_id, so re-seeing it in the active segment is a harmless
    # overwrite, not a double count.
    rotated_path = active_path.with_name(f"{active_path.stem}.1{active_path.suffix}")
    segment_paths = [p for p in (rotated_path, active_path) if p.exists()]
    if not segment_paths:
        return {"ips": 0.0, "dr": 0.0, "n": 0, "ess": 0.0}

    selects: dict[str, dict[str, Any]] = {}
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # Strict streaming decode shared with the WAL replay path (OPE-VAL-1): rejects NaN/Infinity
    # and any malformed non-final line as corruption, and discards only a truncated final line
    # (FR-24) instead of a blanket `except Exception: continue` that would silently drop it.
    for path_obj in segment_paths:
        for entry in _read_journal(path_obj):
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

        # Reconstruct the candidate's exploration term from the persisted, unscaled radius: the
        # logged bonus alone cannot be rescaled once logged_alpha is 0 (OPE-1).
        radius = np.array(s_entry["radius"], dtype=np.float64)
        bonus_cand = candidate_static.alpha * radius

        # The logged score already includes each arm's quota penalty. Candidate evaluation changes
        # only the exploration term, so it must not use the selected arm's units for every arm.
        score_cand = logged_score + bonus_cand - logged_bonus

        # Active arms come from the persisted structural mask, not the logged policy: an active
        # arm's policy can underflow to 0 under a zero epsilon_floor or a heavy quota penalty,
        # which would otherwise misclassify it as inactive (OPE-2).
        active = np.array(s_entry["active"], dtype=bool)
        k_act = int(np.sum(active))

        if k_act == 0:
            continue

        # Subtract the max *before* dividing by tau, mirroring linucb.py's online path
        # (R4-SF-1): a tiny tau can only push a logit to -inf, never to inf - inf = NaN.
        max_score = np.max(np.where(active, score_cand, -np.inf))
        exp_logits = np.where(active, np.exp((score_cand - max_score) / candidate_static.tau), 0.0)
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
