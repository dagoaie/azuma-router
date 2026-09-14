"""Configuration models and TOML parser.

Strict parsing according to contract §4.1 and ADR-0012, ADR-0017.
"""

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from azuma_router.errors import ConfigError
from azuma_router.types import StaticConfig


@dataclass(frozen=True)
class WindowConfig:
    """Window configuration for quota limiting."""

    kind: str  # "rolling" | "fixed_reset"
    capacity: float
    duration_s: int | None = None
    period: str | None = None  # "daily" | "weekly"
    reset_anchor: str | None = None


@dataclass(frozen=True)
class ArmConfig:
    """Configuration for a single routing arm."""

    model_id: str
    provider: str
    billing_mode: str  # "subscription" | "per_token"
    price_in_per_1m: float
    price_out_per_1m: float
    units_per_request: float
    units_per_1k_tokens: float
    barrier_mu: float
    windows: tuple[WindowConfig, ...]

    @property
    def arm_id(self) -> str:
        return f"{self.model_id}@{self.provider}"


@dataclass(frozen=True)
class BanditConfig:
    """Global bandit hyperparameters and runtime limits."""

    seed: int
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
    pending_ttl_s: int
    snapshot_dir: str
    snapshot_every_updates: int
    snapshot_every_s: int
    decision_log_path: str
    decision_log_max_bytes: int


@dataclass(frozen=True)
class Config:
    """Complete immutable configuration loaded from TOML."""

    schema_version: int
    bandit: BanditConfig
    arms: tuple[ArmConfig, ...]


def _check_type(val: Any, expected_type: type | tuple[type, ...], field_path: str) -> None:
    if not isinstance(val, expected_type):
        raise ConfigError(field_path, f"Expected {expected_type}, got {type(val).__name__}")


def _req_dict_val(
    data_dict: dict[str, Any],
    prefix: str,
    name: str,
    types: type | tuple[type, ...],
) -> Any:
    full_path = f"{prefix}.{name}"
    if name not in data_dict:
        raise ConfigError(full_path, f"Missing mandatory field '{full_path}'")
    val = data_dict[name]
    if types is float and isinstance(val, int) and not isinstance(val, bool):
        val = float(val)
    _check_type(val, types, full_path)
    return val


def _parse_bandit_config(b_data: dict[str, Any]) -> BanditConfig:
    def req_b(name: str, types: type | tuple[type, ...]) -> Any:
        return _req_dict_val(b_data, "bandit", name, types)

    seed = req_b("seed", int)
    if seed < 0:
        raise ConfigError("bandit.seed", "seed must be >= 0")

    k_max = req_b("k_max", int)
    if not (1 <= k_max <= 256):
        raise ConfigError("bandit.k_max", f"k_max must be between 1 and 256, got {k_max}")

    w_max = req_b("w_max", int)
    if not (1 <= w_max <= 8):
        raise ConfigError("bandit.w_max", f"w_max must be between 1 and 8, got {w_max}")

    d = req_b("d", int)
    if d != 32:
        raise ConfigError("bandit.d", f"d must be 32 for featurizer v1, got {d}")

    ridge_lambda = req_b("ridge_lambda", float)
    if ridge_lambda <= 0:
        raise ConfigError("bandit.ridge_lambda", "ridge_lambda must be > 0")

    alpha = req_b("alpha", float)
    if alpha < 0:
        raise ConfigError("bandit.alpha", "alpha must be >= 0")

    tau = req_b("tau", float)
    if tau <= 0:
        raise ConfigError("bandit.tau", "tau must be > 0")

    epsilon_floor = req_b("epsilon_floor", float)
    if not (0.0 <= epsilon_floor <= 0.5):
        raise ConfigError("bandit.epsilon_floor", "epsilon_floor must be in [0, 0.5]")

    w_quality = req_b("w_quality", float)
    if w_quality < 0:
        raise ConfigError("bandit.w_quality", "w_quality must be >= 0")

    w_cost = req_b("w_cost", float)
    if w_cost < 0:
        raise ConfigError("bandit.w_cost", "w_cost must be >= 0")

    w_latency = req_b("w_latency", float)
    if w_latency < 0:
        raise ConfigError("bandit.w_latency", "w_latency must be >= 0")

    cost_budget_usd = req_b("cost_budget_usd", float)
    if cost_budget_usd <= 0:
        raise ConfigError("bandit.cost_budget_usd", "cost_budget_usd must be > 0")

    latency_budget_s = req_b("latency_budget_s", float)
    if latency_budget_s <= 0:
        raise ConfigError("bandit.latency_budget_s", "latency_budget_s must be > 0")

    pending_ttl_s = req_b("pending_ttl_s", int)
    if pending_ttl_s <= 0:
        raise ConfigError("bandit.pending_ttl_s", "pending_ttl_s must be > 0")

    snapshot_dir = req_b("snapshot_dir", str)
    snapshot_every_updates = req_b("snapshot_every_updates", int)
    if snapshot_every_updates <= 0:
        raise ConfigError("bandit.snapshot_every_updates", "snapshot_every_updates must be > 0")

    snapshot_every_s = req_b("snapshot_every_s", int)
    if snapshot_every_s <= 0:
        raise ConfigError("bandit.snapshot_every_s", "snapshot_every_s must be > 0")

    decision_log_path = req_b("decision_log_path", str)
    decision_log_max_bytes = req_b("decision_log_max_bytes", int)
    if decision_log_max_bytes <= 0:
        raise ConfigError("bandit.decision_log_max_bytes", "decision_log_max_bytes must be > 0")

    return BanditConfig(
        seed=seed,
        k_max=k_max,
        w_max=w_max,
        d=d,
        ridge_lambda=ridge_lambda,
        alpha=alpha,
        tau=tau,
        epsilon_floor=epsilon_floor,
        w_quality=w_quality,
        w_cost=w_cost,
        w_latency=w_latency,
        cost_budget_usd=cost_budget_usd,
        latency_budget_s=latency_budget_s,
        pending_ttl_s=pending_ttl_s,
        snapshot_dir=snapshot_dir,
        snapshot_every_updates=snapshot_every_updates,
        snapshot_every_s=snapshot_every_s,
        decision_log_path=decision_log_path,
        decision_log_max_bytes=decision_log_max_bytes,
    )


def _parse_window_config(w_data: Any, w_prefix: str) -> WindowConfig:
    if not isinstance(w_data, dict):
        raise ConfigError(w_prefix, "Window entry must be a dictionary")

    kind = w_data.get("kind")
    if kind not in ("rolling", "fixed_reset"):
        msg = f"kind must be 'rolling' or 'fixed_reset', got {kind!r}"
        raise ConfigError(f"{w_prefix}.kind", msg)

    cap = w_data.get("capacity")
    if cap is None:
        raise ConfigError(f"{w_prefix}.capacity", "Missing mandatory field 'capacity'")
    if isinstance(cap, int) and not isinstance(cap, bool):
        cap = float(cap)
    if not isinstance(cap, float) or cap <= 0:
        raise ConfigError(f"{w_prefix}.capacity", "capacity must be a float > 0")

    duration_s: int | None = None
    period: str | None = None
    reset_anchor: str | None = None

    if kind == "rolling":
        duration_s = w_data.get("duration_s")
        if duration_s is None or not isinstance(duration_s, int) or duration_s <= 0:
            msg = "rolling window requires duration_s int > 0"
            raise ConfigError(f"{w_prefix}.duration_s", msg)
    elif kind == "fixed_reset":
        period = w_data.get("period")
        if period not in ("daily", "weekly"):
            msg = "fixed_reset window period must be 'daily' or 'weekly'"
            raise ConfigError(f"{w_prefix}.period", msg)
        reset_anchor = w_data.get("reset_anchor")
        if not reset_anchor or not isinstance(reset_anchor, str):
            msg = "fixed_reset window requires reset_anchor RFC 3339 string"
            raise ConfigError(f"{w_prefix}.reset_anchor", msg)
        try:
            datetime.fromisoformat(reset_anchor.replace("Z", "+00:00"))
        except ValueError as exc:
            msg = f"Invalid RFC 3339 timestamp: {reset_anchor!r}"
            raise ConfigError(f"{w_prefix}.reset_anchor", msg) from exc

    return WindowConfig(
        kind=kind,
        capacity=cap,
        duration_s=duration_s,
        period=period,
        reset_anchor=reset_anchor,
    )


def _parse_arm_config(a_data: Any, index: int, w_max: int, seen_arm_ids: set[str]) -> ArmConfig:
    prefix = f"arms[{index}]"
    if not isinstance(a_data, dict):
        raise ConfigError(prefix, "Arm entry must be a dictionary")

    model_id = _req_dict_val(a_data, prefix, "model_id", str)
    if not model_id or "@" in model_id:
        raise ConfigError(f"{prefix}.model_id", "model_id cannot be empty or contain '@'")

    provider = _req_dict_val(a_data, prefix, "provider", str)
    if not provider or "@" in provider:
        raise ConfigError(f"{prefix}.provider", "provider cannot be empty or contain '@'")

    arm_id = f"{model_id}@{provider}"
    if arm_id in seen_arm_ids:
        raise ConfigError(prefix, f"Duplicate arm_id: {arm_id}")
    seen_arm_ids.add(arm_id)

    billing_mode = _req_dict_val(a_data, prefix, "billing_mode", str)
    if billing_mode not in ("subscription", "per_token"):
        raise ConfigError(
            f"{prefix}.billing_mode",
            f"billing_mode must be 'subscription' or 'per_token', got {billing_mode!r}",
        )

    price_in_per_1m = _req_dict_val(a_data, prefix, "price_in_per_1m", float)
    if price_in_per_1m < 0:
        raise ConfigError(f"{prefix}.price_in_per_1m", "price_in_per_1m must be >= 0")
    if billing_mode == "subscription" and price_in_per_1m != 0.0:
        msg = "price_in_per_1m must be 0 for subscription arms"
        raise ConfigError(f"{prefix}.price_in_per_1m", msg)

    price_out_per_1m = _req_dict_val(a_data, prefix, "price_out_per_1m", float)
    if price_out_per_1m < 0:
        raise ConfigError(f"{prefix}.price_out_per_1m", "price_out_per_1m must be >= 0")
    if billing_mode == "subscription" and price_out_per_1m != 0.0:
        msg = "price_out_per_1m must be 0 for subscription arms"
        raise ConfigError(f"{prefix}.price_out_per_1m", msg)

    units_per_request = _req_dict_val(a_data, prefix, "units_per_request", float)
    if units_per_request < 0:
        raise ConfigError(f"{prefix}.units_per_request", "units_per_request must be >= 0")

    units_per_1k_tokens = _req_dict_val(a_data, prefix, "units_per_1k_tokens", float)
    if units_per_1k_tokens < 0:
        raise ConfigError(f"{prefix}.units_per_1k_tokens", "units_per_1k_tokens must be >= 0")

    barrier_mu = _req_dict_val(a_data, prefix, "barrier_mu", float)
    if barrier_mu < 0:
        raise ConfigError(f"{prefix}.barrier_mu", "barrier_mu must be >= 0")
    if billing_mode == "per_token" and barrier_mu != 0.0:
        raise ConfigError(f"{prefix}.barrier_mu", "barrier_mu must be 0.0 for per_token arms")

    raw_windows = a_data.get("windows", [])
    if billing_mode == "per_token" and "windows" in a_data and raw_windows:
        raise ConfigError(f"{prefix}.windows", "per_token arms cannot have quota windows")

    if billing_mode == "subscription" and not raw_windows:
        msg = "subscription arms must have at least one window"
        raise ConfigError(f"{prefix}.windows", msg)

    if len(raw_windows) > w_max:
        raise ConfigError(
            f"{prefix}.windows",
            f"Number of windows ({len(raw_windows)}) exceeds w_max ({w_max})",
        )

    window_configs = [
        _parse_window_config(w_data, f"{prefix}.windows[{j}]")
        for j, w_data in enumerate(raw_windows)
    ]

    return ArmConfig(
        model_id=model_id,
        provider=provider,
        billing_mode=billing_mode,
        price_in_per_1m=price_in_per_1m,
        price_out_per_1m=price_out_per_1m,
        units_per_request=units_per_request,
        units_per_1k_tokens=units_per_1k_tokens,
        barrier_mu=barrier_mu,
        windows=tuple(window_configs),
    )


def load_config(path: str | Path) -> Config:
    path_obj = Path(path)
    if not path_obj.exists():
        raise ConfigError("file", f"Configuration file not found: {path_obj}")

    with path_obj.open("rb") as f:
        try:
            data = tomllib.load(f)
        except Exception as e:
            raise ConfigError("toml", f"TOML parse error: {e}") from e

    # schema_version
    if "schema_version" not in data:
        raise ConfigError("schema_version", "Missing mandatory field 'schema_version'")
    schema_ver = data["schema_version"]
    if not isinstance(schema_ver, int) or schema_ver != 1:
        raise ConfigError("schema_version", f"schema_version must be 1, got {schema_ver!r}")

    # [bandit]
    if "bandit" not in data or not isinstance(data["bandit"], dict):
        raise ConfigError("bandit", "Missing mandatory section '[bandit]'")

    bandit_cfg = _parse_bandit_config(data["bandit"])

    # [[arms]]
    if "arms" not in data or not isinstance(data["arms"], list):
        raise ConfigError("arms", "Missing mandatory list '[[arms]]'")

    arms_list = data["arms"]
    if len(arms_list) < 1 or len(arms_list) > bandit_cfg.k_max:
        msg = (
            f"Number of arms must be between 1 and k_max ({bandit_cfg.k_max}), got {len(arms_list)}"
        )
        raise ConfigError("arms", msg)

    seen_arm_ids: set[str] = set()
    arm_configs = [
        _parse_arm_config(a_data, i, bandit_cfg.w_max, seen_arm_ids)
        for i, a_data in enumerate(arms_list)
    ]

    return Config(
        schema_version=schema_ver,
        bandit=bandit_cfg,
        arms=tuple(arm_configs),
    )


def to_static_config(config: Config) -> StaticConfig:
    b = config.bandit
    k_max = b.k_max

    barrier_mu_list: list[float] = []
    is_sub_list: list[bool] = []

    for arm in config.arms:
        barrier_mu_list.append(arm.barrier_mu)
        is_sub_list.append(arm.billing_mode == "subscription")

    # Pad up to k_max
    pad_len = k_max - len(config.arms)
    if pad_len > 0:
        barrier_mu_list.extend([0.0] * pad_len)
        is_sub_list.extend([False] * pad_len)

    return StaticConfig(
        k_max=b.k_max,
        w_max=b.w_max,
        d=b.d,
        ridge_lambda=b.ridge_lambda,
        alpha=b.alpha,
        tau=b.tau,
        epsilon_floor=b.epsilon_floor,
        w_quality=b.w_quality,
        w_cost=b.w_cost,
        w_latency=b.w_latency,
        cost_budget_usd=b.cost_budget_usd,
        latency_budget_s=b.latency_budget_s,
        barrier_mu=tuple(barrier_mu_list),
        is_subscription=tuple(is_sub_list),
    )
