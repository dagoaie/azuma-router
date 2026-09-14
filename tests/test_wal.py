"""Tests for DecisionLog WAL and PendingStore (T16 & T17)."""

import json
from pathlib import Path

import pytest

from azuma_router.config import load_config
from azuma_router.daemon import Router
from azuma_router.errors import RpcError
from azuma_router.types import Pending
from azuma_router.wal import DecisionLog, PendingStore


def test_decision_log_append_flush_ac17(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    dlog = DecisionLog(path=log_path, max_bytes=1000)

    # Spy on flush
    flushed = False
    original_flush = dlog.file.flush

    def spy_flush() -> None:
        nonlocal flushed
        original_flush()
        flushed = True

    dlog.file.flush = spy_flush

    entry = {
        "phase": "select",
        "decision_id": "018f6c2e-0000-7000-8000-000000000001",
        "ts": 1757800000.0,
        "state_version": 0,
        "arm_index": 0,
        "arm_id": "gpt-5-codex@openai-codex",
        "propensity": 0.8123,
        "x": [1.0] * 32,
        "policy": [0.8123, 0.1877],
        "score": [0.71, 0.44],
        "exploit": [0.5, 0.3],
        "bonus": [0.21, 0.14],
        "shadow_price": [0.0, 0.0],
        "utilization": {"gpt-5-codex@openai-codex": [0.0, 0.0]},
        "est_units": 1.0,
        "featurizer_version": 1,
    }

    dlog.append(entry)
    assert flushed is True

    # Read back file content
    with log_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["phase"] == "select"
    assert parsed["decision_id"] == "018f6c2e-0000-7000-8000-000000000001"
    dlog.close()


def test_decision_log_rotation(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    dlog = DecisionLog(path=log_path, max_bytes=50)

    select_entry = {
        "phase": "select",
        "decision_id": "018f6c2e-0000-7000-8000-000000000001",
        "ts": 1757800000.0,
        "state_version": 0,
    }

    dlog.append(select_entry)
    assert dlog.needs_rotation is True

    dlog.rotate(carry_forward=[select_entry])

    rotated_path = tmp_path / "decisions.1.jsonl"
    assert rotated_path.exists()
    dlog.close()


def test_pending_store_sweep() -> None:
    store = PendingStore()
    p1 = Pending(
        decision_id="id-1",
        arm_index=0,
        x=[1.0] * 32,
        propensity=0.8,
        ts_select=100.0,
        est_units=1.0,
        ledger_event_id=1,
        state_version=0,
    )
    p2 = Pending(
        decision_id="id-2",
        arm_index=1,
        x=[1.0] * 32,
        propensity=0.2,
        ts_select=200.0,
        est_units=1.0,
        ledger_event_id=2,
        state_version=0,
    )

    store.put(p1)
    store.put(p2)
    assert len(store) == 2

    # Sweep at now = 150.0, ttl = 30.0 -> p1 expired (now - ts = 50 > 30), p2 kept (150 - 200 < 0)
    expired = store.sweep(now=150.0, ttl_s=30.0)
    assert len(expired) == 1
    assert expired[0].decision_id == "id-1"
    assert len(store) == 1
    assert "id-2" in store


def test_journal_determinism_across_runs(tmp_path: Path) -> None:
    from azuma_router.config import load_config
    from azuma_router.daemon import Router

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


def test_expired_ids_eviction_limit_lru(tmp_path: Path) -> None:
    store = PendingStore(max_lru=3)

    store.mark_expired("id-1", 100.0)
    store.mark_expired("id-2", 101.0)
    store.mark_expired("id-3", 102.0)
    store.mark_expired("id-4", 103.0)

    # id-1 (oldest) evicted
    assert not store.is_expired("id-1")
    assert store.is_expired("id-2")
    assert store.is_expired("id-3")
    assert store.is_expired("id-4")

    cfg = load_config(Path("fixtures/config/azuma.valid.toml"))
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()
    router.pending_store._max_lru = 3

    router.pending_store.mark_expired("exp-1", 10.0)
    router.pending_store.mark_expired("exp-2", 20.0)
    router.pending_store.mark_expired("exp-3", 30.0)
    router.pending_store.mark_expired("exp-4", 40.0)

    # exp-4 (newest) returns -32003
    with pytest.raises(RpcError) as exc_4:
        router.report(
            {
                "decision_id": "exp-4",
                "quality": 0.5,
                "cost_usd": 0.0,
                "latency_s": 1.0,
                "actual_input_tokens": 10,
                "actual_output_tokens": 10,
            }
        )
    assert exc_4.value.code == -32003

    # exp-1 (evicted) returns -32002
    with pytest.raises(RpcError) as exc_1:
        router.report(
            {
                "decision_id": "exp-1",
                "quality": 0.5,
                "cost_usd": 0.0,
                "latency_s": 1.0,
                "actual_input_tokens": 10,
                "actual_output_tokens": 10,
            }
        )
    assert exc_1.value.code == -32002

    router.close()
