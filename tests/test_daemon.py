"""Unit and integration tests for Router daemon orchestration and protocol lifecycle."""

import importlib
import io
import json
import math
import shutil
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from azuma_router import daemon as daemon_mod
from azuma_router.config import load_config
from azuma_router.daemon import Router, ShutdownCoordinator, _read_line_bounded, main, serve
from azuma_router.errors import (
    JournalWriteError,
    NumericalIntegrityError,
    RpcError,
    SnapshotWriteError,
)
from azuma_router.quota import QuotaLedger
from azuma_router.wal import PendingStore
from conftest import load_fixture_config_with

# `azuma_router.snapshot` the attribute is the re-exported function, not the module.
snapshot_mod = importlib.import_module("azuma_router.snapshot")

CODEX = "gpt-5-codex@openai-codex"
REQUEST = {
    "task_kind": "code_edit",
    "input_tokens": 100,
    "expected_output_tokens": 50,
    "has_code": True,
    "has_tools": False,
    "has_images": False,
    "agent_depth": 1,
    "diff_lines": 10,
    "priority": "normal",
}


def _report_params(decision_id: str) -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "quality": 0.8,
        "cost_usd": 0.0,
        "latency_s": 1.0,
        "actual_input_tokens": 100,
        "actual_output_tokens": 50,
    }


def _booted_router(
    tmp_path: Path, clock_fn: Callable[[], float] | None = None, **overrides: object
) -> Router:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    router = Router(
        config=cfg,
        snapshot_dir=tmp_path / ".azuma",
        decision_log_path=tmp_path / "decisions.jsonl",
        clock_fn=clock_fn,
    )
    router.boot()
    for name, value in overrides.items():
        setattr(router.dlog, name, value)
    return router


def _codex_utilization(router: Router) -> float:
    return float(router.health({})["arms"][CODEX]["utilization"][0])


def test_router_pending_selected_before_a_later_update_survives_snapshot_restart(
    tmp_path: Path,
) -> None:
    """A live pending must be replayed even when the snapshot state_version has moved past it."""
    router = _booted_router(tmp_path)
    first = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    second = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    router.report(_report_params(second))
    router.snapshot({})
    utilization_before = _codex_utilization(router)
    router.close()

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["pending"] == 1
    assert _codex_utilization(restarted) == pytest.approx(utilization_before)
    assert restarted.report(_report_params(first))["applied"] is True
    assert _codex_utilization(restarted) == pytest.approx(utilization_before)
    restarted.close()


def test_router_pending_selected_before_a_later_update_survives_rotation_restart(
    tmp_path: Path,
) -> None:
    """Same scenario when the snapshot is the one forced by WAL rotation."""
    router = _booted_router(tmp_path, max_bytes=2500)
    first = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    for _ in range(4):
        later = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
        router.report(_report_params(later))
    assert (tmp_path / "decisions.1.jsonl").exists()
    utilization_before = _codex_utilization(router)
    router.close()

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["pending"] == 1
    assert restarted.report(_report_params(first))["applied"] is True
    assert _codex_utilization(restarted) == pytest.approx(utilization_before)
    restarted.close()


def test_router_report_and_select_validation_error_handling(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel_res = router.select({"request": req})
    d_id = sel_res["decision_id"]

    # Initial state snapshots for byte-identical checks
    orig_a = np.array(router.state.A, copy=True)
    orig_b = np.array(router.state.b, copy=True)
    orig_ver = router.state_version
    orig_wal = l_path.read_bytes()
    orig_ledger_dump = router.ledger.dump()

    # Reject booleans & non-finite floats in report
    invalid_reports = [
        ("quality", True),
        ("quality", float("nan")),
        ("quality", float("inf")),
        ("quality", -0.1),
        ("quality", 1.1),
        ("cost_usd", True),
        ("cost_usd", float("nan")),
        ("cost_usd", float("inf")),
        ("cost_usd", -1.0),
        ("latency_s", True),
        ("latency_s", float("nan")),
        ("latency_s", float("inf")),
        ("latency_s", -0.5),
        ("actual_input_tokens", True),
        ("actual_input_tokens", 3.14),
        ("actual_input_tokens", -5),
        ("actual_input_tokens", 2**31),
        ("actual_output_tokens", True),
        ("actual_output_tokens", 2.718),
        ("actual_output_tokens", -1),
        ("actual_output_tokens", 2**31),
    ]

    for field_name, bad_val in invalid_reports:
        rep_params = {
            "decision_id": d_id,
            "quality": 0.8,
            "cost_usd": 0.0,
            "latency_s": 1.0,
            "actual_input_tokens": 100,
            "actual_output_tokens": 50,
        }
        rep_params[field_name] = bad_val
        with pytest.raises(RpcError) as exc_info:
            router.report(rep_params)
        assert exc_info.value.code == -32602
        assert exc_info.value.data is not None
        assert exc_info.value.data["field"] == field_name

    # Check state, ledger, state_version, and WAL file remain byte-identical
    np.testing.assert_array_equal(np.array(router.state.A), orig_a)
    np.testing.assert_array_equal(np.array(router.state.b), orig_b)
    assert router.state_version == orig_ver
    assert l_path.read_bytes() == orig_wal
    assert router.ledger.dump() == orig_ledger_dump

    # Reject booleans & non-finite/invalid numeric fields in select RoutingRequest
    invalid_selects = [
        ("input_tokens", True),
        ("input_tokens", float("nan")),
        ("input_tokens", float("inf")),
        ("input_tokens", -10),
        ("input_tokens", 2**31),
        ("expected_output_tokens", True),
        ("expected_output_tokens", float("nan")),
        ("expected_output_tokens", float("inf")),
        ("expected_output_tokens", 2**31),
        ("agent_depth", True),
        ("agent_depth", float("nan")),
        ("agent_depth", -1),
        ("agent_depth", 2**31),
        ("diff_lines", True),
        ("diff_lines", float("inf")),
        ("diff_lines", -5),
        ("diff_lines", 2**31),
    ]

    for field_name, bad_val in invalid_selects:
        bad_req = dict(req)
        bad_req[field_name] = bad_val
        with pytest.raises(RpcError) as exc_info:
            router.select({"request": bad_req})
        assert exc_info.value.code == -32001
        assert exc_info.value.data is not None
        assert exc_info.value.data["field"] == field_name

    router.close()


def test_router_transactional_error_handling_on_wal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }

    def bad_append(entry: dict) -> None:
        raise JournalWriteError("Disk full error")

    assert router.dlog is not None
    monkeypatch.setattr(router.dlog, "append", bad_append)

    with pytest.raises(JournalWriteError):
        router.select({"request": req})

    assert len(router.pending_store) == 0
    assert router.state_version == 0
    assert sum(len(q) for q in router.ledger.events.values()) == 0

    monkeypatch.undo()
    sel_res = router.select({"request": req})
    d_id = sel_res["decision_id"]
    assert len(router.pending_store) == 1

    monkeypatch.setattr(router.dlog, "append", bad_append)
    rep_params = {
        "decision_id": d_id,
        "quality": 0.8,
        "cost_usd": 0.0,
        "latency_s": 1.0,
        "actual_input_tokens": 100,
        "actual_output_tokens": 50,
    }
    with pytest.raises(JournalWriteError):
        router.report(rep_params)

    assert router.state_version == 0
    assert len(router.pending_store) == 1
    assert not router.pending_store.is_applied(d_id)

    monkeypatch.undo()

    def bad_snapshot(params: dict) -> dict:
        raise RuntimeError("Snapshot failed")

    monkeypatch.setattr(router, "snapshot", bad_snapshot)
    exit_code = router.close()
    assert exit_code == 4


def test_router_replay_ledger_rebuild_parity(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    r1 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r1.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 1000,
        "expected_output_tokens": 500,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel1 = r1.select({"request": req})
    d_id = sel1["decision_id"]
    if r1.dlog is not None:
        r1.dlog.close()

    r2 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r2.boot()
    assert len(r2.pending_store) == 1

    rep_res = r2.report(
        {
            "decision_id": d_id,
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 2.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )
    assert rep_res["applied"] is True

    l_path_b = tmp_path / "baseline.jsonl"
    s_dir_b = tmp_path / "base_snap"
    r_base = Router(
        config=cfg, snapshot_dir=s_dir_b, decision_log_path=l_path_b, clock_fn=mock_clock
    )
    r_base.boot()
    sel_b = r_base.select({"request": req})
    r_base.report(
        {
            "decision_id": sel_b["decision_id"],
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 2.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )

    u2 = r2.ledger.utilization(curr_time)
    ub = r_base.ledger.utilization(curr_time)
    np.testing.assert_array_equal(u2, ub)

    r2.close()
    r_base.close()


def test_router_restart_after_snapshot_ledger_dedup_and_correction(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    r1 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r1.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 1000,
        "expected_output_tokens": 500,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel1 = r1.select({"request": req})
    r1.report(
        {
            "decision_id": sel1["decision_id"],
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 2.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )
    r1.snapshot({})
    assert r1.state_version == 1

    sel2 = r1.select({"request": req})
    d2_id = sel2["decision_id"]

    if r1.dlog is not None:
        r1.dlog.close()

    s_dir_b = tmp_path / "base_snap"
    l_path_b = tmp_path / "base_decisions.jsonl"
    rb = Router(config=cfg, snapshot_dir=s_dir_b, decision_log_path=l_path_b, clock_fn=mock_clock)
    rb.boot()
    sb1 = rb.select({"request": req})
    rb.report(
        {
            "decision_id": sb1["decision_id"],
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 2.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )
    rb.snapshot({})
    sb2 = rb.select({"request": req})
    rb.report(
        {
            "decision_id": sb2["decision_id"],
            "quality": 0.85,
            "cost_usd": 0.0,
            "latency_s": 1.5,
            "actual_input_tokens": 1200,
            "actual_output_tokens": 400,
        }
    )

    r2 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r2.boot()
    assert len(r2.pending_store) == 1
    assert r2.state_version == 1

    rep2 = r2.report(
        {
            "decision_id": d2_id,
            "quality": 0.85,
            "cost_usd": 0.0,
            "latency_s": 1.5,
            "actual_input_tokens": 1200,
            "actual_output_tokens": 400,
        }
    )
    assert rep2["applied"] is True
    assert rep2["state_version"] == 2

    u2 = r2.ledger.utilization(curr_time)
    ub = rb.ledger.utilization(curr_time)
    np.testing.assert_array_equal(u2, ub)

    r2.close()
    rb.close()


def test_router_report_ttl_sweep_and_expired_decision_error(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel1 = router.select({"request": req})
    d1 = sel1["decision_id"]

    curr_time = 5000.0

    with pytest.raises(RpcError) as exc_info:
        router.report(
            {
                "decision_id": d1,
                "quality": 0.8,
                "cost_usd": 0.0,
                "latency_s": 1.0,
                "actual_input_tokens": 100,
                "actual_output_tokens": 50,
            }
        )
    assert exc_info.value.code == -32003
    assert exc_info.value.data is not None
    assert exc_info.value.data["expired_at"] == 4600.0

    assert int(router.state.n_pulls[0]) == 0
    assert int(router.state.n_pulls[1]) == 0

    sel2 = router.select({"request": req})
    d2 = sel2["decision_id"]
    curr_time = 9000.0

    router.select({"request": req})

    with pytest.raises(RpcError) as exc_info2:
        router.report(
            {
                "decision_id": d2,
                "quality": 0.8,
                "cost_usd": 0.0,
                "latency_s": 1.0,
                "actual_input_tokens": 100,
                "actual_output_tokens": 50,
            }
        )
    assert exc_info2.value.code == -32003
    assert exc_info2.value.data is not None
    assert exc_info2.value.data["expired_at"] == 8600.0

    router.close()


def test_router_select_no_arm_available_preserves_state(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "d.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    orig_a = np.array(router.state.A, copy=True)
    orig_b = np.array(router.state.b, copy=True)
    orig_rng = np.array(router.state.rng_key, copy=True)
    orig_pulls = np.array(router.state.n_pulls, copy=True)

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }

    with pytest.raises(RpcError) as exc_info:
        router.select({"request": req, "allowed_arms": []})

    assert exc_info.value.code == -32004

    np.testing.assert_array_equal(np.array(router.state.A), orig_a)
    np.testing.assert_array_equal(np.array(router.state.b), orig_b)
    np.testing.assert_array_equal(np.array(router.state.rng_key), orig_rng)
    np.testing.assert_array_equal(np.array(router.state.n_pulls), orig_pulls)

    router.close()


def test_router_snapshot_before_ready_and_bounded_framing(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "d.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)

    with pytest.raises(RpcError) as exc_info:
        router.snapshot({})
    assert exc_info.value.code == -32005

    large_stream = io.BytesIO(b"x" * 2000000 + b"\nnext_line\n")
    line_str, is_over = _read_line_bounded(large_stream, limit=1048576)
    assert line_str is None
    assert is_over is True

    line2, is_over2 = _read_line_bounded(large_stream, limit=1048576)
    assert line2 == "next_line\n"
    assert is_over2 is False


def test_router_stderr_json_logging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "d.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel1 = router.select({"request": req})
    d1 = sel1["decision_id"]
    router.report(
        {
            "decision_id": d1,
            "quality": 0.8,
            "cost_usd": 0.0,
            "latency_s": 1.0,
            "actual_input_tokens": 100,
            "actual_output_tokens": 50,
        }
    )

    router.report(
        {
            "decision_id": d1,
            "quality": 0.8,
            "cost_usd": 0.0,
            "latency_s": 1.0,
            "actual_input_tokens": 100,
            "actual_output_tokens": 50,
        }
    )

    curr_time = 990.0
    router.select({"request": req})

    router.snapshot({})
    router.close()

    captured = capsys.readouterr()
    err_lines = [line for line in captured.err.splitlines() if line.strip()]

    events_found = set()
    for line in err_lines:
        data = json.loads(line)
        assert "ts" in data
        assert "level" in data
        assert "event" in data
        events_found.add(data["event"])

    assert "ready" in events_found
    assert "snapshot_written" in events_found
    assert "report_duplicate" in events_found
    assert "clock_skew" in events_found


# Snapshot Replay Ledger Idempotency Tests
def test_snapshot_replay_ledger_idempotency_scenarios(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    # Scenario A: select -> snapshot -> restart -> report gives utilization identical to no-restart
    r_nr = Router(
        snapshot_dir=tmp_path / "snap_nr",
        decision_log_path=tmp_path / "dec_nr.jsonl",
        config=cfg,
        clock_fn=mock_clock,
    )
    r_nr.boot()
    req = {
        "task_kind": "code_edit",
        "input_tokens": 1000,
        "expected_output_tokens": 500,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }
    sel_nr = r_nr.select({"request": req})
    r_nr.snapshot({})
    r_nr.report(
        {
            "decision_id": sel_nr["decision_id"],
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 1.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )
    u_no_restart = r_nr.ledger.utilization(curr_time)
    r_nr.close()

    r1 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r1.boot()
    sel1 = r1.select({"request": req})
    d1_id = sel1["decision_id"]
    r1.snapshot({})
    r1.dlog.close()

    # Restart
    r2 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r2.boot()
    r2.report(
        {
            "decision_id": d1_id,
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 1.0,
            "actual_input_tokens": 1000,
            "actual_output_tokens": 500,
        }
    )
    u_restart = r2.ledger.utilization(curr_time)
    np.testing.assert_array_equal(u_restart, u_no_restart)
    r2.close()

    # Scenario B: select -> snapshot -> select -> restart records exactly one new event
    s_dir2 = tmp_path / ".azuma2"
    l_path2 = tmp_path / "decisions2.jsonl"
    r3 = Router(config=cfg, snapshot_dir=s_dir2, decision_log_path=l_path2, clock_fn=mock_clock)
    r3.boot()
    r3.select({"request": req})
    r3.snapshot({})
    r3.select({"request": req})
    events_before_restart = sum(len(q) for q in r3.ledger.events.values())
    r3.dlog.close()

    r4 = Router(config=cfg, snapshot_dir=s_dir2, decision_log_path=l_path2, clock_fn=mock_clock)
    r4.boot()
    events_after_restart = sum(len(q) for q in r4.ledger.events.values())
    assert events_after_restart == events_before_restart
    r4.close()

    # Scenario C: restart twice in a row is stable
    r5 = Router(config=cfg, snapshot_dir=s_dir2, decision_log_path=l_path2, clock_fn=mock_clock)
    r5.boot()
    events_after_second_restart = sum(len(q) for q in r5.ledger.events.values())
    assert events_after_second_restart == events_before_restart
    r5.close()


def _snapshot_bytes(tmp_path: Path) -> bytes | None:
    state_file = tmp_path / ".azuma" / "snapshot" / "state.json"
    return state_file.read_bytes() if state_file.exists() else None


def test_select_never_takes_the_cadence_snapshot_but_report_does(tmp_path: Path) -> None:
    clock = [1_000_000.0]
    router = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    clock[0] += router.config.bandit.snapshot_every_s + 1  # FR-28 time trigger is due
    before = _snapshot_bytes(tmp_path)

    d_id = router.select({"request": REQUEST})["decision_id"]
    assert _snapshot_bytes(tmp_path) == before, "select is read-only w.r.t. snapshots"

    router.report(_report_params(d_id))
    assert _snapshot_bytes(tmp_path) != before
    router.close()


def _booted_router_with_long_pending_ttl(tmp_path: Path, clock: list[float]) -> Router:
    """Fixture config with pending_ttl_s longer than every quota window (2 x weekly)."""
    cfg = load_fixture_config_with(tmp_path, **{"pending_ttl_s = 3600": "pending_ttl_s = 2000000"})
    router = Router(
        config=cfg,
        snapshot_dir=tmp_path / ".azuma",
        decision_log_path=tmp_path / "decisions.jsonl",
        clock_fn=lambda: clock[0],
    )
    router.boot()
    return router


def test_report_after_units_left_every_window_but_within_ttl_is_applied(tmp_path: Path) -> None:
    """P3-MF-1: pruning the quota windows must not break a still-live pending."""
    clock = [1_000_000.0]
    router = _booted_router_with_long_pending_ttl(tmp_path, clock)
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    clock[0] += 1_300_000.0  # past 2 x weekly, inside pending_ttl_s
    assert _codex_utilization(router) == 0.0

    result = router.report(_report_params(d_id))

    assert result == {"applied": True, "reward": result["reward"], "state_version": 1}
    assert router.report(_report_params(d_id))["reason"] == "duplicate"


def _boot_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str]]:
    router = Router(
        config=load_config(Path("fixtures/config/azuma.valid.toml")),
        snapshot_dir=tmp_path / ".azuma",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    with pytest.raises(SystemExit) as exit_info:
        router.boot()
    events = [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()]
    return int(exit_info.value.code or 0), events


def test_boot_refuses_learned_updates_in_the_rotated_wal_without_a_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P3-MF-2: the no-snapshot guard must scan the rotated segment, not only the active one."""
    router = _booted_router(tmp_path)
    router.report(_report_params(router.select({"request": REQUEST})["decision_id"]))
    router.dlog.close()
    (tmp_path / "decisions.jsonl").replace(tmp_path / "decisions.1.jsonl")
    (tmp_path / "decisions.jsonl").write_text("")
    shutil.rmtree(tmp_path / ".azuma", ignore_errors=True)

    code, events = _boot_exit_code(tmp_path, capsys)

    assert code == 3
    assert "wal_inconsistent" in events


def _snapshot_meta_path(tmp_path: Path) -> Path:
    router = _booted_router(tmp_path)
    router.close()
    return tmp_path / ".azuma" / "snapshot" / "state.json"


def test_boot_exits_3_on_an_incompatible_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snapshot from another schema is an operator problem, not a traceback (R4-MF-2)."""
    meta_path = _snapshot_meta_path(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = meta["schema_version"] + 1
    meta_path.write_text(json.dumps(meta))

    code, events = _boot_exit_code(tmp_path, capsys)

    assert code == 3
    assert "incompatible_snapshot" in events


def test_boot_exits_3_when_the_snapshot_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta_path = _snapshot_meta_path(tmp_path)
    meta_path.chmod(0o000)
    try:
        code, events = _boot_exit_code(tmp_path, capsys)
    finally:
        meta_path.chmod(0o644)

    assert code == 3
    assert "snapshot_read_failed" in events


def test_boot_exits_3_when_the_journal_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = tmp_path / "decisions.jsonl"
    journal.write_text("")
    journal.chmod(0o000)
    try:
        code, events = _boot_exit_code(tmp_path, capsys)
    finally:
        journal.chmod(0o644)

    assert code == 3
    assert "replay_failed" in events


def test_boot_does_not_swallow_a_corrupt_line_while_checking_for_a_missing_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "decisions.jsonl").write_text('{"state_version": 1, "phase": "repo\n{}\n')

    code, events = _boot_exit_code(tmp_path, capsys)

    assert code == 3
    assert "replay_failed" in events


def test_select_stays_finite_at_the_largest_accepted_quota_coefficients_and_tokens(
    tmp_path: Path,
) -> None:
    """P3-MF-5: whatever load_config accepts must survive the float32 JAX boundary."""
    largest = str(2**31 - 1)
    cfg = load_fixture_config_with(
        tmp_path,
        **{
            "units_per_request = 1.0": f"units_per_request = {largest}",
            "units_per_1k_tokens = 0.0\nbarrier_mu = 0.5": (
                f"units_per_1k_tokens = {largest}\nbarrier_mu = 0.5"
            ),
            "capacity = 300.0": f"capacity = {largest}",
        },
    )
    router = Router(config=cfg, snapshot_dir=tmp_path / ".azuma", decision_log_path=tmp_path / "d")
    router.boot()
    request = {**REQUEST, "input_tokens": 2**31 - 1, "expected_output_tokens": 2**31 - 1}

    result = router.select({"request": request})

    assert result["arm"]["arm_id"] in {CODEX, "qwen3-coder@openrouter"}
    assert router.report(_report_params(result["decision_id"]))["applied"] is True
    assert math.isfinite(_codex_utilization(router))


def test_report_on_an_evicted_expired_id_is_unknown_not_expired(tmp_path: Path) -> None:
    router = _booted_router(tmp_path)
    router.pending_store = PendingStore(max_lru=3)
    for n, ts in enumerate((10.0, 20.0, 30.0, 40.0), start=1):
        router.pending_store.mark_expired(f"exp-{n}", ts)

    with pytest.raises(RpcError) as newest:
        router.report(_report_params("exp-4"))
    assert newest.value.code == -32003

    with pytest.raises(RpcError) as evicted:
        router.report(_report_params("exp-1"))
    assert evicted.value.code == -32002
    router.close()


def _normalized_journal(journal: Path) -> list[str]:
    normalized = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        data["ts"] = 0.0
        data.pop("ts_select", None)
        data["decision_id"] = "fixed-uuid"
        normalized.append(json.dumps(data, sort_keys=True))
    return normalized


def test_journal_determinism_across_runs(tmp_path: Path) -> None:
    def run_simulation(run_dir: Path) -> list[str]:
        router = _booted_router(run_dir)
        d_ids = [router.select({"request": REQUEST})["decision_id"] for _ in range(25)]
        for d_id in d_ids:
            router.report(_report_params(d_id))
        router.close()
        return _normalized_journal(run_dir / "decisions.jsonl")

    assert run_simulation(tmp_path / "run1") == run_simulation(tmp_path / "run2")


def _select_ids(journal: Path) -> set[str]:
    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    return {entry["decision_id"] for entry in entries if entry["phase"] == "select"}


def test_wal_rotation_carries_live_pendings_without_duplicating_quota(tmp_path: Path) -> None:
    router = _booted_router(tmp_path, max_bytes=2500)
    d_ids = [
        router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
        for _ in range(6)
    ]
    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists()
    carried = _select_ids(tmp_path / "decisions.jsonl") & _select_ids(rotated_path)
    assert carried, "rotation carries live selects forward, so some select lives in both segments"
    utilization_before = _codex_utilization(router)
    router.close()

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["pending"] == len(d_ids)
    assert _codex_utilization(restarted) == pytest.approx(utilization_before)
    for d_id in d_ids:
        assert restarted.report(_report_params(d_id))["applied"] is True
    assert _codex_utilization(restarted) == pytest.approx(utilization_before)
    restarted.close()


def _expire_all_pendings(clock: list[float], ttl_s: float = 3600.0) -> None:
    clock[0] += ttl_s + 1.0


def test_report_on_an_expired_id_still_rotates_the_journal(tmp_path: Path) -> None:
    """An expire line is an append like any other: it must trigger rotation (P3-SF-1)."""
    clock = [1_700_000_000.0]
    router = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    router.dlog.max_bytes = 1  # type: ignore[union-attr]
    _expire_all_pendings(clock)

    with pytest.raises(RpcError) as excinfo:
        router.report(_report_params(d_id))
    assert excinfo.value.code == -32003

    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists(), "the expire append pushed the journal past max_bytes"
    assert d_id in _select_ids(rotated_path)
    assert _select_ids(tmp_path / "decisions.jsonl") == set(), "an expired select is not carried"
    router.close()

    restarted = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    assert restarted.health({})["pending"] == 0
    restarted.close()


def test_select_that_fails_validation_still_rotates_after_expiring(tmp_path: Path) -> None:
    """Expiries journaled by a rejected select must not wait for the next success (R4-SF-2)."""
    clock = [1_700_000_000.0]
    router = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    router.dlog.max_bytes = 1  # type: ignore[union-attr]
    _expire_all_pendings(clock)

    with pytest.raises(RpcError) as excinfo:
        router.select({"request": "not-an-object"})
    assert excinfo.value.code == -32602

    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists(), "the expire append pushed the journal past max_bytes"
    assert d_id in _select_ids(rotated_path)
    assert _select_ids(tmp_path / "decisions.jsonl") == set(), "an expired select is not carried"
    router.close()


def test_snapshot_rpc_rotates_after_journaling_expiries(tmp_path: Path) -> None:
    """The explicit snapshot RPC journals expiries and rotates like the implicit paths (P3-SF-1)."""
    clock = [1_700_000_000.0]
    router = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    d_ids = {
        router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
        for _ in range(2)
    }
    router.dlog.max_bytes = 1  # type: ignore[union-attr]
    _expire_all_pendings(clock)

    assert router.snapshot({})["expired_pending"] == 2

    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists(), "the expire appends pushed the journal past max_bytes"
    assert _select_ids(rotated_path) == d_ids
    assert _select_ids(tmp_path / "decisions.jsonl") == set()
    router.close()


def test_wal_report_appended_after_rotation_is_reapplied_on_restart(tmp_path: Path) -> None:
    router = _booted_router(tmp_path, max_bytes=2500)
    d_ids = [
        router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
        for _ in range(4)
    ]
    assert (tmp_path / "decisions.1.jsonl").exists()
    assert router.dlog is not None
    router.dlog.max_bytes = 10**9  # freeze rotation: the report must not snapshot again
    snapshot_version = router.health({})["state_version"]
    router.report(_report_params(d_ids[0]))  # lands in the active segment, after the snapshot
    assert router.health({})["state_version"] == snapshot_version + 1
    router.dlog.close()  # crash: no shutdown snapshot

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["state_version"] == snapshot_version + 1
    assert restarted.report(_report_params(d_ids[0]))["applied"] is False
    assert restarted.health({})["pending"] == len(d_ids) - 1
    restarted.close()


def test_verbose_select_masked_arm_diagnostics_null(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": 50,
        "has_code": True,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 1,
        "diff_lines": 10,
        "priority": "normal",
    }

    # Only allow arm 0 -> arm 1 is masked (policy == 0)
    res = router.select(
        {"request": req, "allowed_arms": ["gpt-5-codex@openai-codex"], "verbose": True}
    )
    diag = res["diagnostics"]

    # Masked arm 1 must have null (None) for score, exploit, bonus, shadow_price
    masked_id = "qwen3-coder@openrouter"
    assert diag["score"][masked_id] is None
    assert diag["exploit"][masked_id] is None
    assert diag["bonus"][masked_id] is None
    assert diag["shadow_price"][masked_id] is None

    # Serialization with allow_nan=False MUST succeed
    serialized = json.dumps(res, allow_nan=False)
    parsed = json.loads(serialized)
    assert parsed["diagnostics"]["score"][masked_id] is None

    # WAL line keeps sentinel (-1e9) for score
    lines = l_path.read_text(encoding="utf-8").splitlines()
    wal_entry = json.loads(lines[0])
    assert wal_entry["score"][1] == -1e9

    router.close()


# Non-finite Score Handling Tests
def _force_nan_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes select_jit report NaN scores while leaving the sampled arm untouched."""
    real_select_jit = daemon_mod.select_jit

    def nan_select_jit(state: Any, x: Any, qv: Any, static: Any) -> tuple[Any, Any, Any, Any]:
        state_next, arm_index, propensity, diag = real_select_jit(state, x, qv, static)
        return state_next, arm_index, propensity, diag._replace(score=np.full(static.k_max, np.nan))

    monkeypatch.setattr(daemon_mod, "select_jit", nan_select_jit)


def test_select_non_finite_score_raises_numerical_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _booted_router(tmp_path)
    _force_nan_scores(monkeypatch)

    orig_ver = router.state_version
    with pytest.raises(NumericalIntegrityError):
        router.select({"request": REQUEST})

    assert router.state_version == orig_ver
    assert len(router.pending_store) == 0
    assert (tmp_path / "decisions.jsonl").read_text(encoding="utf-8") == ""

    router.close()


# Journal Write Error Stdio Test
@pytest.mark.parametrize(
    "boot_error",
    [
        JournalWriteError("wal locked"),
        NumericalIntegrityError("nan"),
        SnapshotWriteError("read-only filesystem"),
    ],
)
def test_main_exits_4_when_boot_raises_a_fatal_integrity_error(
    monkeypatch: pytest.MonkeyPatch, boot_error: Exception
) -> None:
    def failing_boot(self: Router) -> dict[str, object]:
        raise boot_error

    monkeypatch.setattr(Router, "boot", failing_boot)
    monkeypatch.setattr(
        sys, "argv", ["azuma-router", "--config", "fixtures/config/azuma.valid.toml"]
    )
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 4
    assert "boot" in stderr.getvalue()


def _serve_one_request(
    router: Router, method: str, params: dict[str, object]
) -> tuple[int | None, list[dict[str, object]]]:
    """Serves a single request until exit; returns (exit code, stderr events)."""
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    stdin, stdout, stderr = io.StringIO(json.dumps(request) + "\n"), io.StringIO(), io.StringIO()
    with pytest.raises(SystemExit) as exit_info:
        serve(stdin, stdout, stderr, router, exit_on_eof=False)
    events = [json.loads(line) for line in stderr.getvalue().splitlines()]
    return exit_info.value.code, events


def test_serve_exits_4_on_journal_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _booted_router(tmp_path)

    def bad_append(entry: dict[str, object]) -> None:
        raise JournalWriteError("Write failed")

    monkeypatch.setattr(router.dlog, "append", bad_append)

    code, events = _serve_one_request(router, "select", {"request": REQUEST})

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["journal_write_error"]


def test_serve_exits_4_before_journaling_a_report_whose_ledger_event_is_missing(
    tmp_path: Path,
) -> None:
    """A pending without its ledger event is corrupt state: fail-stop, never -32603 post-append."""
    router = _booted_router(tmp_path)
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    router.ledger = QuotaLedger(router.config)

    code, events = _serve_one_request(router, "report", _report_params(d_id))

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["ledger_integrity_error"]
    journal = _normalized_journal(tmp_path / "decisions.jsonl")
    assert all('"report"' not in line for line in journal)


def test_serve_exits_4_when_select_hits_a_non_finite_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    router = _booted_router(tmp_path)
    _force_nan_scores(monkeypatch)
    capsys.readouterr()  # drop boot logs

    code, events = _serve_one_request(router, "select", {"request": REQUEST})

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["numerical_integrity_error"]
    assert "non_finite" not in capsys.readouterr().err, "serve owns the fatal log; no double log"
    assert _snapshot_bytes(tmp_path) is None, "corrupt state must never be snapshotted"


def test_serve_exits_4_when_the_rotation_snapshot_after_a_select_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """P3-MF-3: once select committed WAL/ledger/pending, a persistence fault cannot be -32603."""
    router = _booted_router(tmp_path, max_bytes=1)

    def failing_write_snapshot(*_args: object, **_kwargs: object) -> tuple[Path, int]:
        raise SnapshotWriteError("snapshot write failed: disk full")

    monkeypatch.setattr("azuma_router.daemon.write_snapshot", failing_write_snapshot)
    capsys.readouterr()

    code, events = _serve_one_request(router, "select", {"request": REQUEST})

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["snapshot_write_error"]
    assert "shutdown_snapshot_failed" not in capsys.readouterr().err


def test_serve_exits_4_when_a_result_contains_a_non_finite_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _booted_router(tmp_path)
    monkeypatch.setattr(router, "health", lambda params: {"state_version": float("inf")})

    code, events = _serve_one_request(router, "health", {})

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["numerical_integrity_error"]


def _rpc_line(req_id: int, method: str, params: dict[str, object]) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}) + "\n"


def test_a_signal_while_idle_closes_and_exits_immediately(tmp_path: Path) -> None:
    """Nothing is in flight between requests, so the handler may close synchronously (R4-MF-1)."""
    router = _booted_router(tmp_path)
    stderr = io.StringIO()
    shutdown = ShutdownCoordinator(router, stderr)

    with pytest.raises(SystemExit) as exit_info:
        shutdown.on_signal(signal.SIGTERM, None)

    assert exit_info.value.code == 0
    assert (tmp_path / ".azuma" / "snapshot").exists(), "shutdown snapshot written"
    events = [json.loads(line)["event"] for line in stderr.getvalue().splitlines()]
    assert events == ["signal_received"]


def test_a_signal_during_report_is_deferred_until_the_request_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shutdown snapshot taken mid-update would persist a half-applied step (R4-MF-1)."""
    router = _booted_router(tmp_path)
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    stdout, stderr = io.StringIO(), io.StringIO()
    shutdown = ShutdownCoordinator(router, stderr)
    real_update_jit = daemon_mod.update_jit

    def update_jit_then_signal(*args: Any, **kwargs: Any) -> Any:
        shutdown.on_signal(signal.SIGTERM, None)  # arrives while report() is mid-flight
        return real_update_jit(*args, **kwargs)

    monkeypatch.setattr(daemon_mod, "update_jit", update_jit_then_signal)
    stdin = io.StringIO(_rpc_line(1, "report", _report_params(d_id)) + _rpc_line(2, "health", {}))

    with pytest.raises(SystemExit) as exit_info:
        serve(stdin, stdout, stderr, router, exit_on_eof=False, shutdown=shutdown)

    assert exit_info.value.code == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1], "the in-flight report is answered, nothing after it"
    assert replies[0]["result"]["applied"] is True
    events = [json.loads(line)["event"] for line in stderr.getvalue().splitlines()]
    assert events == ["signal_received", "shutdown_deferred"]

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["pending"] == 0
    assert restarted.health({})["state_version"] == 1
    restarted.close()


def test_an_unexpected_failure_after_the_report_append_is_fatal_and_replayed_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuing with an advanced state_version and stale tensors would make the next
    snapshot lie and replay skip a durable report (P5-MF-1)."""
    router = _booted_router(tmp_path)
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]

    def backend_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(daemon_mod, "update_jit", backend_failure)
    code, events = _serve_one_request(router, "report", _report_params(d_id))
    monkeypatch.undo()

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["journal_apply_error"]
    assert not (tmp_path / ".azuma" / "snapshot").exists(), "no snapshot of an uncertain state"

    restarted = _booted_router(tmp_path)
    assert restarted.health({})["state_version"] == 1
    assert restarted.health({})["pending"] == 0
    assert float(restarted.state.n_pulls[0]) == 1.0, "the durable report was re-applied"
    restarted.close()


def test_restart_after_a_kill_before_the_first_snapshot_replays_from_the_initial_state(
    tmp_path: Path,
) -> None:
    """A fresh daemon killed within its first snapshot cadence must come back (no false
    `wal_inconsistent`): with no rotation, the journal is complete from state_version 1."""
    router = _booted_router(tmp_path)
    router.report(
        _report_params(router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"])
    )
    router.dlog.close()  # type: ignore[union-attr]  # SIGKILL: no shutdown snapshot

    restarted = _booted_router(tmp_path)

    assert restarted.health({})["state_version"] == 1
    assert float(restarted.state.n_pulls[0]) == 1.0
    restarted.close()


def test_boot_exits_3_when_replayed_reports_are_not_contiguous_after_the_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gap in state_version means updates were lost between snapshot and journal."""
    router = _booted_router(tmp_path)
    for _ in range(2):
        d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
        router.report(_report_params(d_id))
    router.dlog.close()  # type: ignore[union-attr]
    journal = tmp_path / "decisions.jsonl"
    kept = [
        line
        for line in journal.read_text().splitlines()
        if not (json.loads(line)["phase"] == "report" and json.loads(line)["state_version"] == 1)
    ]
    journal.write_text("\n".join(kept) + "\n")

    code, events = _boot_exit_code(tmp_path, capsys)

    assert code == 3
    assert "replay_failed" in events


def test_a_signal_during_a_fatal_close_is_ignored_and_never_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM landing while `close(with_snapshot=False)` runs must not start a second close
    that takes the very snapshot the fatal path forbids (P5-MF-2)."""
    router = _booted_router(tmp_path)
    d_id = router.select({"request": REQUEST, "allowed_arms": [CODEX]})["decision_id"]
    stderr = io.StringIO()
    shutdown = ShutdownCoordinator(router, stderr)

    def backend_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(daemon_mod, "update_jit", backend_failure)
    real_close, close_calls = router.close, []

    def counting_close(*, with_snapshot: bool = True) -> int:
        close_calls.append(with_snapshot)
        return real_close(with_snapshot=with_snapshot)

    assert router.dlog is not None
    real_dlog_close = router.dlog.close

    def dlog_close_with_signal_landing_inside() -> None:
        shutdown.on_signal(signal.SIGTERM, None)  # mid-close, after the snapshot decision
        real_dlog_close()

    monkeypatch.setattr(router, "close", counting_close)
    monkeypatch.setattr(router.dlog, "close", dlog_close_with_signal_landing_inside)
    request = {"jsonrpc": "2.0", "id": 1, "method": "report", "params": _report_params(d_id)}
    stdin = io.StringIO(json.dumps(request) + "\n")

    with pytest.raises(SystemExit) as exit_info:
        serve(stdin, io.StringIO(), stderr, router, exit_on_eof=False, shutdown=shutdown)

    assert exit_info.value.code == 4
    assert close_calls == [False]
    assert not (tmp_path / ".azuma" / "snapshot").exists()
    events = [json.loads(line)["event"] for line in stderr.getvalue().splitlines()]
    assert events == ["journal_apply_error", "signal_received"]


def test_main_exits_3_when_the_config_file_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unreadable = tmp_path / "azuma.toml"
    unreadable.write_text('schema_version = "1"\n')
    unreadable.chmod(0o000)
    monkeypatch.setattr(sys, "argv", ["azuma-router", "--config", str(unreadable)])
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 3
    assert [json.loads(line)["event"] for line in stderr.getvalue().splitlines()] == [
        "config_error"
    ]


def test_boot_exits_3_when_the_snapshot_ledger_is_malformed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snapshot whose ledger does not decode is incompatible, not a KeyError (P5-MF-3)."""
    meta_path = _snapshot_meta_path(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta["ledger"] = {"events": {CODEX: [{"bogus": 1}]}, "next_event_id": 0}
    meta_path.write_text(json.dumps(meta))

    code, events = _boot_exit_code(tmp_path, capsys)

    assert code == 3
    assert "incompatible_snapshot" in events


def test_per_token_ledger_events_are_pruned_once_no_pending_can_correct_them(
    tmp_path: Path,
) -> None:
    """Per-token arms have no windows but their events must still expire after pending_ttl_s
    or every select leaks into memory and the snapshot forever (P5-MF-4)."""
    per_token_arm = "qwen3-coder@openrouter"
    clock = [1_700_000_000.0]
    router = _booted_router(tmp_path, clock_fn=lambda: clock[0])
    for _ in range(5):
        d_id = router.select({"request": REQUEST, "allowed_arms": [per_token_arm]})["decision_id"]
        router.report(_report_params(d_id))
        clock[0] += 2 * 3600  # twice pending_ttl_s: no live pending can reference older events

    assert len(router.ledger.dump()["events"][per_token_arm]) == 1
    router.close()


def test_serve_exits_4_when_the_snapshot_backend_fails_with_a_non_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A safetensors failure is as fatal as a disk failure: exit 4 with the previous snapshot
    intact, never a -32603 that leaves the daemon serving without snapshots (P6-RR-1)."""
    router = _booted_router(tmp_path)
    router.snapshot({})
    previous_meta = _snapshot_meta_path_of(tmp_path).read_text()

    def backend_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated safetensors backend failure")

    monkeypatch.setattr(snapshot_mod, "save_file", backend_failure)

    code, events = _serve_one_request(router, "snapshot", {})

    assert code == 4
    assert [e["event"] for e in events if e["level"] == "error"] == ["snapshot_write_error"]
    assert _snapshot_meta_path_of(tmp_path).read_text() == previous_meta


def _snapshot_meta_path_of(tmp_path: Path) -> Path:
    return tmp_path / ".azuma" / "snapshot" / "state.json"


def _main_exit_with_sigterm_after(
    monkeypatch: pytest.MonkeyPatch, fatal_event: str
) -> tuple[int, list[str]]:
    """Runs main() and delivers a real SIGTERM right after `fatal_event` is logged: the
    narrowest window in which a signal could replace a fatal exit code with 0 (P6-RR-2)."""
    real_log_event = daemon_mod._log_event

    def log_event_then_signal(stream: Any, level: str, event: str, **fields: Any) -> None:
        real_log_event(stream, level, event, **fields)
        if event == fatal_event:
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(daemon_mod, "_log_event", log_event_then_signal)
    monkeypatch.setattr(sys, "argv", ["azuma-router", "--config", "azuma.toml"])
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    events = [json.loads(line)["event"] for line in stderr.getvalue().splitlines()]
    return int(exit_info.value.code or 0), events


def test_a_signal_right_after_an_unrecoverable_boot_failure_keeps_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_fixture_config_with(tmp_path)  # writes tmp_path/azuma.toml with relative paths
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".azuma").mkdir()
    (tmp_path / ".azuma" / "decisions.1.jsonl").write_text("")  # rotated segment, no snapshot

    code, events = _main_exit_with_sigterm_after(monkeypatch, "wal_inconsistent")

    assert code == 3
    assert events == ["wal_inconsistent", "signal_received"]


def test_a_signal_right_after_a_fatal_boot_failure_keeps_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_fixture_config_with(tmp_path)
    monkeypatch.chdir(tmp_path)

    def failing_boot(self: Router) -> dict[str, object]:
        raise NumericalIntegrityError("simulated corrupt state")

    monkeypatch.setattr(Router, "boot", failing_boot)

    code, events = _main_exit_with_sigterm_after(monkeypatch, "numerical_integrity_error")

    assert code == 4
    assert events == ["numerical_integrity_error", "signal_received"]
