"""Unit and integration tests for Router daemon orchestration and protocol lifecycle."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config
from azuma_router.daemon import Router, _read_line_bounded, serve
from azuma_router.errors import JournalWriteError, NumericalIntegrityError, RpcError


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


# WAL Rotation Carry-Forward Tests
def test_wal_rotation_carry_forward_and_dedup(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    # Tiny max_bytes so every select triggers rotation check
    r = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    r.config = cfg
    r.boot()
    r.dlog.max_bytes = 200

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

    # N selects across multiple rotations without reports
    d_ids = []
    for _ in range(5):
        s = r.select({"request": req})
        d_ids.append(s["decision_id"])

    r.dlog.close()

    # Restart daemon from disk
    r_restarted = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    r_restarted.boot()

    # Every pending must still be reportable without -32002
    for d_id in d_ids:
        res = r_restarted.report(
            {
                "decision_id": d_id,
                "quality": 0.8,
                "cost_usd": 0.0,
                "latency_s": 1.0,
                "actual_input_tokens": 100,
                "actual_output_tokens": 50,
            }
        )
        assert res["applied"] is True

    r_restarted.close()


# Diagnostics Masked Arm Null Tests
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
def test_select_non_finite_score_raises_numerical_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    from azuma_router import daemon as daemon_mod

    orig_select_jit = daemon_mod.select_jit

    from typing import Any

    def mock_select_jit(state: Any, x: Any, qv: Any, static: Any) -> tuple[Any, Any, Any, Any]:
        st_next, arm_i, prop, diag = orig_select_jit(state, x, qv, static)
        nan_score = np.array([float("nan")] * static.k_max)
        new_diag = diag._replace(score=nan_score)
        return st_next, arm_i, prop, new_diag

    monkeypatch.setattr(daemon_mod, "select_jit", mock_select_jit)

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

    orig_ver = router.state_version
    with pytest.raises(NumericalIntegrityError):
        router.select({"request": req})

    assert router.state_version == orig_ver
    assert len(router.pending_store) == 0
    assert not l_path.exists() or len(l_path.read_text(encoding="utf-8").strip()) == 0

    router.close()


# Journal Write Error Stdio Test
def test_serve_exits_4_on_journal_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req_json = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "select",
        "params": {
            "request": {
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
        },
    }

    stdin = io.StringIO(json.dumps(req_json) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    def bad_append(entry: dict) -> None:
        raise JournalWriteError("Write failed")

    assert router.dlog is not None
    monkeypatch.setattr(router.dlog, "append", bad_append)

    exited_code = None

    def mock_exit(code: int) -> None:
        nonlocal exited_code
        exited_code = code
        raise SystemExit(code)

    import sys

    monkeypatch.setattr(sys, "exit", mock_exit)

    with pytest.raises(SystemExit):
        serve(stdin, stdout, stderr, router, exit_on_eof=False)

    assert exited_code == 4
