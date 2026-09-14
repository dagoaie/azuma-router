"""End-to-end subprocess lifecycle test for azuma_router.daemon (T25 & AC-25)."""

import json
import subprocess
import sys
from pathlib import Path


def test_daemon_e2e_subprocess_lifecycle(tmp_path: Path) -> None:
    # Prepare a config with snapshot_dir inside tmp_path
    cfg_text = f"""schema_version = 1

[bandit]
seed = 42
k_max = 32
w_max = 4
d = 32
ridge_lambda = 1.0
alpha = 1.0
tau = 0.05
epsilon_floor = 0.02
w_quality = 1.0
w_cost = 0.3
w_latency = 0.2
cost_budget_usd = 0.50
latency_budget_s = 30.0
pending_ttl_s = 3600
snapshot_dir = "{(tmp_path / '.azuma').as_posix()}"
snapshot_every_updates = 100
snapshot_every_s = 30
decision_log_path = "{(tmp_path / '.azuma' / 'decisions.jsonl').as_posix()}"
decision_log_max_bytes = 268435456

[[arms]]
model_id = "gpt-5-codex"
provider = "openai-codex"
billing_mode = "subscription"
price_in_per_1m = 0.0
price_out_per_1m = 0.0
units_per_request = 1.0
units_per_1k_tokens = 0.0
barrier_mu = 0.5

  [[arms.windows]]
  kind = "rolling"
  duration_s = 18000
  capacity = 300.0

[[arms]]
model_id = "qwen3-coder"
provider = "openrouter"
billing_mode = "per_token"
price_in_per_1m = 0.30
price_out_per_1m = 1.20
units_per_request = 0.0
units_per_1k_tokens = 0.0
barrier_mu = 0.0
"""
    config_file = tmp_path / "azuma.valid.toml"
    config_file.write_text(cfg_text, encoding="utf-8")

    cmd = [sys.executable, "-m", "azuma_router.daemon", "--config", str(config_file)]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    assert proc.stdin is not None

    # Read ready notification
    ready_line = proc.stdout.readline()
    ready_msg = json.loads(ready_line)
    assert ready_msg["method"] == "ready"
    assert ready_msg["params"]["restored_from_snapshot"] is False

    # Send select
    sel_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "select",
        "params": {
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
        },
    }
    proc.stdin.write(json.dumps(sel_req) + "\n")
    proc.stdin.flush()

    sel_line = proc.stdout.readline()
    sel_res = json.loads(sel_line)
    assert sel_res["id"] == 1
    decision_id = sel_res["result"]["decision_id"]

    # Send report
    rep_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "report",
        "params": {
            "decision_id": decision_id,
            "quality": 0.9,
            "cost_usd": 0.0,
            "latency_s": 11.2,
            "actual_input_tokens": 1900,
            "actual_output_tokens": 640,
        },
    }
    proc.stdin.write(json.dumps(rep_req) + "\n")
    proc.stdin.flush()

    rep_line = proc.stdout.readline()
    rep_res = json.loads(rep_line)
    assert rep_res["id"] == 2
    assert rep_res["result"]["state_version"] == 1

    # Send health
    h_req = {"jsonrpc": "2.0", "id": 3, "method": "health", "params": {}}
    proc.stdin.write(json.dumps(h_req) + "\n")
    proc.stdin.flush()

    h_line = proc.stdout.readline()
    h_res = json.loads(h_line)
    assert h_res["result"]["state_version"] == 1
    assert h_res["result"]["pending"] == 0

    # SIGTERM process for clean shutdown & snapshot
    proc.terminate()
    proc.wait(timeout=5)
    assert proc.returncode == 0

    # Relauch process -> should restore from snapshot
    proc2 = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc2.stdout is not None
    assert proc2.stdin is not None

    ready_line2 = proc2.stdout.readline()
    ready_msg2 = json.loads(ready_line2)
    assert ready_msg2["method"] == "ready"
    assert ready_msg2["params"]["restored_from_snapshot"] is True
    assert ready_msg2["params"]["state_version"] == 1
    assert ready_msg2["params"]["warmup_ms"] < 5000.0

    proc2.terminate()
    proc2.wait(timeout=5)
    assert proc2.returncode == 0
