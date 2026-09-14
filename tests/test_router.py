"""Tests for Router orchestration class (T21 & T22)."""

from pathlib import Path

import pytest

from azuma_router.config import load_config
from azuma_router.daemon import Router
from azuma_router.errors import RpcError


def test_router_lifecycle_select_report(tmp_path: Path) -> None:
    cfg_path = Path("fixtures/config/azuma.valid.toml")
    cfg = load_config(cfg_path)

    # Use custom snapshot and log paths in tmp_path
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    boot_resp = router.boot()
    assert boot_resp["state_version"] == 0
    assert len(boot_resp["arms"]) == 2

    # Select request
    sel_params = {
        "request": {
            "task_kind": "code_edit",
            "input_tokens": 1834,
            "expected_output_tokens": 600,
            "has_code": True,
            "has_tools": True,
            "has_images": False,
            "agent_depth": 2,
            "diff_lines": 120,
            "priority": "normal",
        }
    }
    sel_res = router.select(sel_params)
    assert "decision_id" in sel_res
    decision_id = sel_res["decision_id"]
    assert sel_res["arm"]["arm_id"] == "gpt-5-codex@openai-codex"
    assert sel_res["propensity"] > 0.0

    # Report request
    rep_params = {
        "decision_id": decision_id,
        "quality": 0.9,
        "cost_usd": 0.0,
        "latency_s": 11.2,
        "actual_input_tokens": 1900,
        "actual_output_tokens": 640,
    }
    rep_res = router.report(rep_params)
    assert rep_res["applied"] is True
    assert rep_res["state_version"] == 1

    # Duplicate report
    dup_res = router.report(rep_params)
    assert dup_res["applied"] is False
    assert dup_res["reason"] == "duplicate"

    # Health check
    h_res = router.health({})
    assert h_res["ready"] is True
    assert h_res["state_version"] == 1
    assert h_res["pending"] == 0

    router.close()


def test_router_subscription_cost_error(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    sel_res = router.select(
        {
            "request": {
                "task_kind": "code_edit",
                "input_tokens": 100,
                "expected_output_tokens": None,
                "has_code": False,
                "has_tools": False,
                "has_images": False,
                "agent_depth": 0,
                "diff_lines": None,
                "priority": "normal",
            }
        }
    )
    decision_id = sel_res["decision_id"]

    # Subscription arm with cost_usd > 0 must raise -32602
    with pytest.raises(RpcError) as exc_info:
        router.report(
            {
                "decision_id": decision_id,
                "quality": 0.9,
                "cost_usd": 0.05,
                "latency_s": 1.0,
                "actual_input_tokens": 100,
                "actual_output_tokens": 50,
            }
        )
    assert exc_info.value.code == -32602
    assert exc_info.value.data == {"field": "cost_usd"}
    router.close()


def test_select_verbose_false_and_allowed_arms(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": None,
        "has_code": False,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 0,
        "diff_lines": None,
        "priority": "normal",
    }

    # Test verbose=False
    res_no_diag = router.select({"request": req, "verbose": False})
    assert "diagnostics" not in res_no_diag

    # Test allowed_arms filtering
    res_allowed = router.select({"request": req, "allowed_arms": ["qwen3-coder@openrouter"]})
    assert res_allowed["arm"]["arm_id"] == "qwen3-coder@openrouter"

    # Test allowed_arms unknown arm
    with pytest.raises(RpcError) as exc_info:
        router.select({"request": req, "allowed_arms": ["nonexistent@provider"]})
    assert exc_info.value.code == -32602
    assert exc_info.value.data == {"field": "allowed_arms[nonexistent@provider]"}

    # Test no_arm_available (-32004) when allowed_arms is empty
    with pytest.raises(RpcError) as exc_info2:
        router.select({"request": req, "allowed_arms": []})
    assert exc_info2.value.code == -32004
    assert "masked_reason" in exc_info2.value.data

    router.close()


def test_report_validation_errors(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    # Unknown decision_id
    with pytest.raises(RpcError) as exc1:
        router.report({"decision_id": "nonexistent-uuid", "quality": 0.5})
    assert exc1.value.code == -32002

    # Invalid quality (> 1.0)
    with pytest.raises(RpcError) as exc2:
        router.report({"decision_id": "some-id", "quality": 1.5})
    assert exc2.value.code == -32602
    assert exc2.value.data == {"field": "quality"}

    # Invalid cost_usd (< 0)
    with pytest.raises(RpcError) as exc3:
        router.report({"decision_id": "some-id", "quality": 0.5, "cost_usd": -0.1})
    assert exc3.value.code == -32602
    assert exc3.value.data == {"field": "cost_usd"}

    router.close()


def test_allowed_arms_non_string_item(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "d.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": None,
        "has_code": False,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 0,
        "diff_lines": None,
        "priority": "normal",
    }

    with pytest.raises(RpcError) as exc:
        router.select({"request": req, "allowed_arms": [123]})
    assert exc.value.code == -32602

    router.close()


def test_duplicate_report_after_reboot(tmp_path: Path) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"

    r1 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    r1.boot()
    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": None,
        "has_code": False,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 0,
        "diff_lines": None,
        "priority": "normal",
    }
    sel_res = r1.select({"request": req})
    d_id = sel_res["decision_id"]

    rep_params = {
        "decision_id": d_id,
        "quality": 0.8,
        "cost_usd": 0.0,
        "latency_s": 1.0,
        "actual_input_tokens": 100,
        "actual_output_tokens": 50,
    }
    rep_res1 = r1.report(rep_params)
    assert rep_res1["applied"] is True
    r1.close()

    # Reboot router
    r2 = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    r2.boot()

    # Second report should be duplicate even after reboot
    rep_res2 = r2.report(rep_params)
    assert rep_res2["applied"] is False
    assert rep_res2["reason"] == "duplicate"
    r2.close()


def test_clock_skew_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    current_time = 1000.0

    def mock_clock() -> float:
        nonlocal current_time
        return current_time

    router = Router(
        config=cfg,
        snapshot_dir=tmp_path / ".azuma",
        decision_log_path=tmp_path / "d.jsonl",
        clock_fn=mock_clock,
    )
    router.boot()

    req = {
        "task_kind": "code_edit",
        "input_tokens": 100,
        "expected_output_tokens": None,
        "has_code": False,
        "has_tools": False,
        "has_images": False,
        "agent_depth": 0,
        "diff_lines": None,
        "priority": "normal",
    }
    router.select({"request": req})

    # Clock jumps backward 5 seconds
    current_time = 990.0
    router.select({"request": req})

    captured = capsys.readouterr()
    assert "clock_skew" in captured.err
    router.close()
