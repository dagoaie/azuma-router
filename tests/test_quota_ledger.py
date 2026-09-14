"""Tests for QuotaLedger and windowing calculations (T12 & T13)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config
from azuma_router.quota import QuotaLedger


def test_rolling_window_expiry_ac13() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)

    # Arm 0 has rolling window duration_s = 18000 (5h), capacity = 300.0
    arm_id = cfg.arms[0].arm_id
    t0 = 1000.0

    ledger.record(arm_id, 150.0, t0)
    u_early = ledger.utilization(now=t0 + 100.0)
    np.testing.assert_allclose(u_early[0, 0], 0.5, rtol=1e-5)

    # After duration_s + 1, event is expired from the rolling window
    u_late = ledger.utilization(now=t0 + 18001.0)
    assert u_late[0, 0] == 0.0


test_anchor_iso = "2026-09-08T00:00:00Z"
anchor_dt = datetime.fromisoformat(test_anchor_iso.replace("Z", "+00:00"))
anchor_ts = anchor_dt.timestamp()


def test_fixed_reset_weekly_ac13() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    arm_id = cfg.arms[0].arm_id

    # Record 750 units right before next weekly reset (anchor + 7 days - 10s)
    week_s = 7 * 86400.0
    ts_before_reset = anchor_ts + week_s - 10.0

    ledger.record(arm_id, 750.0, ts_before_reset)
    u_before = ledger.utilization(now=ts_before_reset + 1.0)
    np.testing.assert_allclose(u_before[0, 1], 0.5, rtol=1e-5)

    # After reset timestamp (anchor + 7 days + 1s)
    ts_after_reset = anchor_ts + week_s + 1.0
    u_after = ledger.utilization(now=ts_after_reset)
    assert u_after[0, 1] == 0.0


def test_record_and_correct_inplace_ac15() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    arm_id = cfg.arms[0].arm_id

    event_id = ledger.record(arm_id, 2.1, 1000.0)
    assert len(ledger.events[arm_id]) == 1
    assert ledger.events[arm_id][0].units == 2.1

    ledger.correct(event_id, arm_id, 2.54)
    assert len(ledger.events[arm_id]) == 1
    assert ledger.events[arm_id][0].units == 2.54


def test_per_token_arm_zeros() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    # Arm 1 is per_token
    u = ledger.utilization(now=1000.0)
    np.testing.assert_array_equal(u[1], 0.0)


def test_dump_and_load_roundtrip() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    arm_id = cfg.arms[0].arm_id

    ledger.record(arm_id, 50.0, 1000.0)
    ledger.record(arm_id, 100.0, 2000.0)

    dump_data = ledger.dump()
    restored_ledger = QuotaLedger.load(dump_data, cfg)

    u_orig = ledger.utilization(now=2500.0)
    u_rest = restored_ledger.utilization(now=2500.0)
    np.testing.assert_array_equal(u_orig, u_rest)


def test_ledger_event_not_found_error() -> None:
    from azuma_router.errors import LedgerEventNotFoundError

    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)
    with pytest.raises(LedgerEventNotFoundError):
        ledger.correct(99999, "gpt-5-codex@openai-codex", 1.0)
