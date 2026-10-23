"""Subprocess client speaking JSON-RPC 2.0 line by line to `python -m azuma_router.daemon`."""

import json
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

SHUTDOWN_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 30.0


class RpcCallError(Exception):
    """A JSON-RPC error response, surfaced with its code, message and data."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class DaemonProcess:
    """One daemon process: start, call, crash (SIGKILL) or stop (SIGTERM).

    Use it as a context manager: leaving the block always kills and reaps the process, so a
    failing scenario never leaves a daemon or open streams behind.
    """

    def __init__(self, config_path: Path, stderr_path: Path) -> None:
        self._config_path = config_path
        self._stderr_path = stderr_path
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file: TextIO | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self.call_latencies_ms: dict[str, list[float]] = {}

    def __enter__(self) -> "DaemonProcess":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> dict[str, Any]:
        """Launches the daemon and returns the params of its `ready` notification.

        If the daemon never becomes ready, it is killed and reaped before the error propagates.
        """
        try:
            return self._launch()
        except BaseException:
            self.close()
            raise

    def _launch(self) -> dict[str, Any]:
        self._stderr_file = self._stderr_path.open("a", encoding="utf-8")
        command = [sys.executable, "-m", "azuma_router.daemon", "--config", str(self._config_path)]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        threading.Thread(
            target=self._pump_stdout, args=(self._process.stdout,), daemon=True
        ).start()
        ready = json.loads(self._readline())
        if ready.get("method") != "ready":
            raise RuntimeError(f"expected a ready notification, got {ready!r}")
        return dict(ready["params"])

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sends one request and returns its `result`; raises RpcCallError on an error response."""
        self._next_id += 1
        started = time.perf_counter()
        response = self.call_raw(
            json.dumps(
                {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.call_latencies_ms.setdefault(method, []).append(elapsed_ms)
        if "error" in response:
            error = response["error"]
            raise RpcCallError(error["code"], error["message"], error.get("data"))
        return dict(response["result"])

    def call_raw(self, line: str) -> dict[str, Any]:
        """Writes one raw line (a newline is appended) and returns the decoded response."""
        self.write_raw(line + "\n")
        return dict(json.loads(self._readline()))

    def write_raw(self, text: str) -> None:
        """Writes bytes to stdin as-is, without waiting for a response."""
        process = self._require_process()
        assert process.stdin is not None
        process.stdin.write(text)
        process.stdin.flush()

    def read_response(self) -> dict[str, Any]:
        """Reads the next response line (used after `write_raw`)."""
        return dict(json.loads(self._readline()))

    def close(self) -> None:
        """Kills the daemon if it is still alive and releases its streams; safe to repeat."""
        if self._process is None:
            self._close_stderr()
            return
        self._process.kill()
        try:
            self._process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        finally:
            self._release()

    def kill(self) -> int:
        """Crashes the daemon with SIGKILL (no shutdown snapshot) and returns its exit code."""
        return self._stop(signal.SIGKILL)

    def terminate(self) -> int:
        """Stops the daemon with SIGTERM (graceful: shutdown snapshot) and returns its exit code."""
        return self._stop(signal.SIGTERM)

    def _stop(self, sig: signal.Signals) -> int:
        process = self._require_process()
        process.send_signal(sig)
        try:
            exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        finally:
            self._release()
        return exit_code

    def _release(self) -> None:
        process = self._require_process()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        self._close_stderr()
        self._process = None

    def _close_stderr(self) -> None:
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def _pump_stdout(self, stdout: TextIO) -> None:
        """Moves stdout lines onto a queue so reads can time out instead of hanging a test."""
        for line in stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _readline(self) -> str:
        try:
            line = self._lines.get(timeout=READ_TIMEOUT_S)
        except queue.Empty:
            self.close()
            raise TimeoutError(
                f"no response within {READ_TIMEOUT_S}s; see {self._stderr_path}"
            ) from None
        if line is None:
            raise RuntimeError(f"daemon closed stdout; see {self._stderr_path}")
        return line

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("daemon is not running")
        return self._process
