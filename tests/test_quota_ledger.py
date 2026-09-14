"""Tests for QuotaLedger and windowing calculations (T12 & T13)."""

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config
from azuma_router.errors import LedgerEventDuplicateError, LedgerEventNotFoundError
from azuma_router.quota import QuotaLedger

CODEX = "gpt-5-codex@openai-codex"


def _ledger() -> QuotaLedger:
    return QuotaLedger(load_config(Path("fixtures/config/azuma.valid.toml")))


def test_rolling_window_expiry_ac13() -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    ledger = QuotaLedger(cfg)

    # Arm 0 has rolling window duration_s = 18000 (5h), capacity = 300.0
    arm_id = cfg.arms[0].arm_id
    t0 = 1000.0

    ledger.record(ledger.next_event_id(), arm_id, 150.0, t0)
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

    ledger.record(ledger.next_event_id(), arm_id, 750.0, ts_before_reset)
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

    event_id = ledger.next_event_id()
    ledger.record(event_id, arm_id, 2.1, 1000.0)
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

    ledger.record(ledger.next_event_id(), arm_id, 50.0, 1000.0)
    ledger.record(ledger.next_event_id(), arm_id, 100.0, 2000.0)

    dump_data = ledger.dump()
    restored_ledger = QuotaLedger.load(dump_data, cfg)

    u_orig = ledger.utilization(now=2500.0)
    u_rest = restored_ledger.utilization(now=2500.0)
    np.testing.assert_array_equal(u_orig, u_rest)


def test_ledger_event_not_found_error() -> None:
    with pytest.raises(LedgerEventNotFoundError):
        _ledger().correct(99999, CODEX, 1.0)


def test_event_ids_are_allocated_then_recorded_once() -> None:
    ledger = _ledger()
    event_id = ledger.next_event_id()
    assert event_id == 1
    assert not ledger.has_event(event_id)

    ledger.record(event_id, CODEX, 2.0, 1000.0)
    assert ledger.has_event(event_id)
    with pytest.raises(LedgerEventDuplicateError):
        ledger.record(event_id, CODEX, 2.0, 1000.0)
    assert ledger.next_event_id() == 2


def test_recording_a_replayed_event_id_advances_the_allocator() -> None:
    ledger = _ledger()
    ledger.record(7, CODEX, 1.0, 1000.0)  # id minted by a previous process, replayed from the WAL
    assert ledger.next_event_id() == 8


def test_pruned_events_are_forgotten_by_has_event() -> None:
    ledger = _ledger()
    ledger.record(1, CODEX, 1.0, 1000.0)
    beyond_every_window = 1000.0 + 30 * 86400.0
    ledger.utilization(now=beyond_every_window)
    assert not ledger.has_event(1)
    assert ledger.events[CODEX] == deque()


def test_load_restores_event_ids_and_allocator() -> None:
    ledger = _ledger()
    ledger.record(ledger.next_event_id(), CODEX, 1.0, 1000.0)
    ledger.record(ledger.next_event_id(), CODEX, 1.0, 1000.0)

    restored = QuotaLedger.load(ledger.dump(), _ledger().config)
    assert restored.has_event(1) and restored.has_event(2)
    assert restored.next_event_id() == 3
