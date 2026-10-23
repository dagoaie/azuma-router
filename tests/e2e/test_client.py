"""The e2e client must never leave a daemon process or its streams behind, even on failure."""

from pathlib import Path

import pytest

from e2e import client, world
from e2e.client import DaemonProcess

pytestmark = pytest.mark.e2e


def make_daemon(work_dir: Path, *, config_text: str | None = None) -> DaemonProcess:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "azuma.toml"
    config_path.write_text(config_text or world.build_config_toml(work_dir), encoding="utf-8")
    return DaemonProcess(config_path, work_dir / "daemon.stderr.log")


def test_close_stops_a_running_daemon_and_is_idempotent(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    assert daemon.is_running
    daemon.close()
    daemon.close()
    assert not daemon.is_running


def test_leaving_the_context_after_an_error_still_reaps_the_daemon(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    with pytest.raises(ZeroDivisionError), daemon:
        1 / 0  # noqa: B018
    assert not daemon.is_running


def test_a_failed_start_leaves_no_process_behind(tmp_path: Path) -> None:
    daemon = make_daemon(tmp_path, config_text="this is = not [valid toml")
    with pytest.raises(RuntimeError, match="daemon closed stdout"):
        daemon.start()
    assert not daemon.is_running


def test_a_read_timeout_kills_and_reaps_the_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = make_daemon(tmp_path)
    daemon.start()
    monkeypatch.setattr(client, "READ_TIMEOUT_S", 0.05)
    with pytest.raises(TimeoutError):
        daemon.read_response()
    assert not daemon.is_running
