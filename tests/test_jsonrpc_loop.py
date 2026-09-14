"""Tests for stdio JSON-RPC 2.0 loop framing and dispatch (T24 & AC-26, AC-27)."""

import io
import json
from pathlib import Path

import pytest

from azuma_router.config import load_config
from azuma_router.daemon import Router, serve


def test_jsonrpc_loop_serve_stdio(tmp_path: Path) -> None:
    cfg = load_config("fixtures/config/azuma.valid.toml")
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)

    # Serve before boot should reject with -32005 NotReady
    stdin_pre = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"health"}\n')
    stdout_pre = io.StringIO()
    stderr_pre = io.StringIO()

    serve(stdin_pre, stdout_pre, stderr_pre, router, exit_on_eof=False)
    stdout_pre.seek(0)
    out_line = stdout_pre.readline()
    assert "-32005" in out_line

    # Boot router
    router.boot()

    # Batch request rejected with -32600
    stdin_batch = io.StringIO('[{"jsonrpc":"2.0","id":1,"method":"health"}]\n')
    stdout_batch = io.StringIO()
    stderr_batch = io.StringIO()

    serve(stdin_batch, stdout_batch, stderr_batch, router, exit_on_eof=False)
    stdout_batch.seek(0)
    out_batch = stdout_batch.readline()
    assert "-32600" in out_batch

    router.close()


def test_jsonrpc_loop_framing_and_errors(tmp_path: Path) -> None:
    cfg = load_config("fixtures/config/azuma.valid.toml")
    s_dir = tmp_path / ".azuma"
    l_path = tmp_path / "decisions.jsonl"
    router = Router(config=cfg, snapshot_dir=s_dir, decision_log_path=l_path)
    router.boot()

    # 1. Line > 1 MiB -> -32600 line_too_long followed by valid request
    long_line = "a" * (1048576 + 10) + "\n"
    valid_line = '{"jsonrpc":"2.0","id":99,"method":"health"}\n'
    stdin_long = io.StringIO(long_line + valid_line)
    stdout_long = io.StringIO()
    stderr_long = io.StringIO()
    serve(stdin_long, stdout_long, stderr_long, router, exit_on_eof=False)
    out_lines = stdout_long.getvalue().strip().splitlines()
    assert len(out_lines) == 2
    out_long = json.loads(out_lines[0])
    assert out_long["error"]["code"] == -32600
    assert out_long["error"]["data"]["reason"] == "line_too_long"
    out_valid = json.loads(out_lines[1])
    assert out_valid["id"] == 99
    assert out_valid["result"]["ready"] is True

    # 2. Parse error -> -32700
    stdin_parse = io.StringIO("{invalid_json\n")
    stdout_parse = io.StringIO()
    stderr_parse = io.StringIO()
    serve(stdin_parse, stdout_parse, stderr_parse, router, exit_on_eof=False)
    out_parse = json.loads(stdout_parse.getvalue().strip())
    assert out_parse["error"]["code"] == -32700

    # 3. Invalid request (missing jsonrpc 2.0 or method) -> -32600
    stdin_inv = io.StringIO('{"id": 1, "method": "health"}\n')
    stdout_inv = io.StringIO()
    stderr_inv = io.StringIO()
    serve(stdin_inv, stdout_inv, stderr_inv, router, exit_on_eof=False)
    out_inv = json.loads(stdout_inv.getvalue().strip())
    assert out_inv["error"]["code"] == -32600

    # 4. Unknown method -> -32601
    stdin_unk = io.StringIO('{"jsonrpc":"2.0","id":4,"method":"nonexistent"}\n')
    stdout_unk = io.StringIO()
    stderr_unk = io.StringIO()
    serve(stdin_unk, stdout_unk, stderr_unk, router, exit_on_eof=False)
    out_unk = json.loads(stdout_unk.getvalue().strip())
    assert out_unk["error"]["code"] == -32601

    # 5. Notification without id -> process without stdout response
    stdin_notif = io.StringIO('{"jsonrpc":"2.0","method":"snapshot","params":{}}\n')
    stdout_notif = io.StringIO()
    stderr_notif = io.StringIO()
    serve(stdin_notif, stdout_notif, stderr_notif, router, exit_on_eof=False)
    assert stdout_notif.getvalue() == ""

    router.close()


def _booted_router(tmp_path: Path) -> Router:
    router = Router(
        config=load_config("fixtures/config/azuma.valid.toml"),
        snapshot_dir=tmp_path / ".azuma",
        decision_log_path=tmp_path / "decisions.jsonl",
    )
    router.boot()
    return router


def _serve_lines(router: Router, *lines: str) -> list[str]:
    stdout = io.StringIO()
    serve(io.StringIO("".join(f"{line}\n" for line in lines)), stdout, io.StringIO(), router, False)
    return stdout.getvalue().splitlines()


@pytest.mark.parametrize(
    "raw",
    [
        '{"jsonrpc":"2.0","id":NaN,"method":"health"}',
        '{"jsonrpc":"2.0","id":Infinity,"method":"nope"}',
        '{"jsonrpc":"2.0","id":1,"method":"select","params":{"request":{"input_tokens":-Infinity}}}',
    ],
)
def test_non_standard_json_constants_are_a_parse_error_and_never_echoed(
    tmp_path: Path, raw: str
) -> None:
    """P3-MF-4: NaN/Infinity are not JSON; they must neither kill the daemon nor reach stdout."""
    router = _booted_router(tmp_path)

    out = _serve_lines(router, raw, '{"jsonrpc":"2.0","id":7,"method":"health"}')

    assert json.loads(out[0]) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }
    assert json.loads(out[1])["id"] == 7


@pytest.mark.parametrize("bad_id", ["[1]", "{}", "1.5", "true"])
def test_a_request_id_that_is_not_a_string_integer_or_null_is_an_invalid_request(
    tmp_path: Path, bad_id: str
) -> None:
    router = _booted_router(tmp_path)

    out = _serve_lines(router, f'{{"jsonrpc":"2.0","id":{bad_id},"method":"health"}}')

    assert json.loads(out[0]) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32600, "message": "Invalid request", "data": {"reason": "invalid_id"}},
    }


def test_an_explicit_null_id_is_a_request_not_a_notification(tmp_path: Path) -> None:
    """R4-MF-3: JSON-RPC 2.0 distinguishes an absent id (notification) from `"id": null`."""
    router = _booted_router(tmp_path)

    out = _serve_lines(
        router,
        '{"jsonrpc":"2.0","id":null,"method":"health"}',
        '{"jsonrpc":"2.0","id":null,"method":"nope"}',
        '{"jsonrpc":"2.0","method":"health"}',
    )

    assert len(out) == 2, "two null-id requests answered, the notification stays silent"
    assert json.loads(out[0])["id"] is None and "result" in json.loads(out[0])
    assert json.loads(out[1]) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32601, "message": "Method not found: nope"},
    }


@pytest.mark.parametrize("method", ["select", "report", "snapshot", "health"])
@pytest.mark.parametrize("params", ["null", "[]", "1", '"x"'])
def test_params_that_are_not_an_object_are_invalid_params(
    tmp_path: Path, method: str, params: str
) -> None:
    """R4-MF-4: a structurally wrong `params` is the caller's error (-32602), never -32603."""
    router = _booted_router(tmp_path)

    out = _serve_lines(router, f'{{"jsonrpc":"2.0","id":3,"method":"{method}","params":{params}}}')

    assert json.loads(out[0])["error"] == {
        "code": -32602,
        "message": "params must be an object",
        "data": {"field": "params"},
    }


def test_a_pathologically_nested_line_is_a_parse_error_and_the_loop_survives(
    tmp_path: Path,
) -> None:
    """R4-SF-4: the stdlib decoder recurses per nesting level; that must not kill the daemon."""
    router = _booted_router(tmp_path)

    nested = "[" * 100_000 + "]" * 100_000
    out = _serve_lines(router, nested, '{"jsonrpc":"2.0","id":9,"method":"health"}')

    assert json.loads(out[0])["error"]["code"] == -32700
    assert json.loads(out[1])["id"] == 9
