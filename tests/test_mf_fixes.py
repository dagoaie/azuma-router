"""Tests for review pass 1 fixes (MF-1..MF-4, SF-1..SF-8)."""

from pathlib import Path

import numpy as np
import pytest

from azuma_router.config import load_config
from azuma_router.daemon import Router
from azuma_router.errors import RpcError


def test_mf1_non_finite_and_bool_rejection(tmp_path: Path) -> None:
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
        ("actual_output_tokens", True),
        ("actual_output_tokens", 2.718),
        ("actual_output_tokens", -1),
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
        ("expected_output_tokens", True),
        ("expected_output_tokens", float("nan")),
        ("expected_output_tokens", float("inf")),
        ("agent_depth", True),
        ("agent_depth", float("nan")),
        ("agent_depth", -1),
        ("diff_lines", True),
        ("diff_lines", float("inf")),
        ("diff_lines", -5),
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


def test_mf2_faulty_decision_log_transactional(
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

    from azuma_router.errors import JournalWriteError

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


def test_mf3_ledger_rebuild_and_error(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    from azuma_router.errors import LedgerEventNotFoundError
    from azuma_router.quota import QuotaLedger

    ledger = QuotaLedger(cfg)
    with pytest.raises(LedgerEventNotFoundError):
        ledger.correct(99999, "gpt-5-codex@openai-codex", 1.0)

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


def test_mf3_restart_after_snapshot_ledger_correction(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    curr_time = 1000.0

    def mock_clock() -> float:
        return curr_time

    # Run 1: boot, select & report d1, write snapshot (state_version=1),
    # then select d2 (post-snapshot pending)
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

    # Select d2 after snapshot (state_version remains 1)
    sel2 = r1.select({"request": req})
    d2_id = sel2["decision_id"]

    if r1.dlog is not None:
        r1.dlog.close()

    # Baseline router: single uninterrupted run doing sel1, rep1, snap, sel2, rep2
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

    # Run 2: restart r2 from disk (restores snapshot at v1, replays d2 from log)
    r2 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path, clock_fn=mock_clock)
    r2.boot()
    assert len(r2.pending_store) == 1
    assert r2.state_version == 1

    # Now report d2 on r2
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

    # Utilization on r2 and rb must match exactly
    u2 = r2.ledger.utilization(curr_time)
    ub = rb.ledger.utilization(curr_time)
    np.testing.assert_array_equal(u2, ub)

    r2.close()
    rb.close()


def test_mf4_report_ttl_and_expired_ids(tmp_path: Path) -> None:
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


def test_sf1_no_arm_available_preserves_state(tmp_path: Path) -> None:
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


def test_sf3_snapshot_before_ready_and_bounded_framing(tmp_path: Path) -> None:
    import io

    from azuma_router.daemon import _read_line_bounded

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


def test_sf5_jax_import_boundary() -> None:
    import ast

    src_dir = Path("src/azuma_router")
    allowed_jax_modules = {"__init__.py", "types.py", "state.py", "linucb.py"}

    for py_file in src_dir.glob("*.py"):
        if py_file.name in allowed_jax_modules:
            continue

        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        jax_imports = [m for m in imports if "jax" in m]
        assert not jax_imports, f"D18 violation: {py_file.name} imports JAX ({jax_imports})"


def test_sf8_nfr04_stderr_json_logging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

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


def test_nfr06_journal_determinism(tmp_path: Path) -> None:
    import json

    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))

    def run_simulation(out_log: Path) -> list[str]:
        s_dir = tmp_path / f"snap_{out_log.stem}"
        r = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=out_log)
        r.boot()
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
        d_ids = []
        for _ in range(25):
            s = r.select({"request": req})
            d_ids.append(s["decision_id"])
        for d_id in d_ids:
            r.report(
                {
                    "decision_id": d_id,
                    "quality": 0.85,
                    "cost_usd": 0.0,
                    "latency_s": 1.5,
                    "actual_input_tokens": 100,
                    "actual_output_tokens": 50,
                }
            )
        r.close()

        lines = out_log.read_text(encoding="utf-8").splitlines()
        normalized = []
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            data["ts"] = 0.0
            if "ts_select" in data:
                data["ts_select"] = 0.0
            data["decision_id"] = "fixed-uuid"
            normalized.append(json.dumps(data, sort_keys=True))
        return normalized

    run1 = run_simulation(tmp_path / "run1.jsonl")
    run2 = run_simulation(tmp_path / "run2.jsonl")

    assert run1 == run2


def test_nfr07_prohibited_network_pickle_imports() -> None:
    import ast

    src_dir = Path("src/azuma_router")
    banned = {"socket", "pickle", "urllib", "requests", "http", "asyncio", "netrc", "ftplib"}

    for py_file in src_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        violations = imports.intersection(banned)
        assert not violations, f"NFR-07 violation in {py_file.name}: {violations}"
