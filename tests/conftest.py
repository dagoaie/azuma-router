"""Shared fixtures: variants of the valid fixture config written to a temp dir."""

import os
from pathlib import Path

import pytest

from azuma_router.config import Config, load_config

FIXTURE_CONFIG = Path("fixtures/config/azuma.valid.toml")


def load_fixture_config_with(tmp_path: Path, **replacements: str) -> Config:
    """Loads the valid fixture config after replacing each `old` line with `new` exactly once."""
    toml = FIXTURE_CONFIG.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert toml.count(old) == 1, old
        toml = toml.replace(old, new)
    cfg_path = tmp_path / "azuma.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    return load_config(cfg_path)


@pytest.fixture
def fsynced_inodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Records the inode of every descriptor passed to os.fsync, in call order."""
    inodes: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        inodes.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return inodes
