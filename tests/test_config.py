"""Tests for TOML configuration loader and static config projection (T03)."""

from pathlib import Path

import pytest

from azuma_router.config import (
    load_config,
    to_static_config,
)
from azuma_router.errors import ConfigError


def test_load_valid_config() -> None:
    path = Path("fixtures/config/azuma.valid.toml")
    cfg = load_config(path)
    assert cfg.schema_version == 1
    assert len(cfg.arms) == 2
    assert cfg.arms[0].arm_id == "gpt-5-codex@openai-codex"
    assert cfg.arms[1].arm_id == "qwen3-coder@openrouter"
    assert cfg.arms[0].billing_mode == "subscription"
    assert cfg.arms[1].billing_mode == "per_token"
    assert len(cfg.arms[0].windows) == 2
    assert len(cfg.arms[1].windows) == 0


@pytest.mark.parametrize(
    ("filename", "expected_field"),
    [
        ("dup_arm_id.toml", "arms[1]"),
        ("missing_bandit.toml", "bandit"),
        ("sub_without_windows.toml", "arms[0].windows"),
        ("per_token_with_windows.toml", "arms[0].windows"),
        ("per_token_with_mu.toml", "arms[0].barrier_mu"),
        ("sub_with_price_out.toml", "arms[0].price_out_per_1m"),
        ("bad_type.toml", "bandit.seed"),
        ("too_many_arms.toml", "arms"),
        ("too_many_windows.toml", "arms[0].windows"),
        ("bad_period.toml", "arms[0].windows[0].period"),
        ("bad_anchor.toml", "arms[0].windows[0].reset_anchor"),
        ("bool_as_number.toml", "bandit.seed"),
        ("nan_price.toml", "arms[0].price_in_per_1m"),
        ("huge_units_coefficient.toml", "arms[0].units_per_request"),
    ],
)
def test_invalid_configs(filename: str, expected_field: str) -> None:
    path = Path("fixtures/config/invalid") / filename
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert expected_field in exc_info.value.field


def test_to_static_config_padding() -> None:
    path = Path("fixtures/config/azuma.valid.toml")
    cfg = load_config(path)
    static = to_static_config(cfg)

    assert static.k_max == 32
    assert static.w_max == 4
    assert static.d == 32
    assert len(static.barrier_mu) == 32
    assert len(static.is_subscription) == 32

    # Arm 0: subscription, barrier_mu = 0.5
    assert static.barrier_mu[0] == 0.5
    assert static.is_subscription[0] is True

    # Arm 1: per_token, barrier_mu = 0.0
    assert static.barrier_mu[1] == 0.0
    assert static.is_subscription[1] is False

    # Padding slots (2 to 31): 0.0 and False
    assert static.barrier_mu[2:] == (0.0,) * 30
    assert static.is_subscription[2:] == (False,) * 30

    # StaticConfig must be hashable
    assert hash(static) is not None
