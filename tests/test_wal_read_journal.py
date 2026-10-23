"""Tests for the streaming journal reader used by replay and OPE (audit finding #5)."""

import tracemalloc
from pathlib import Path

import pytest

from azuma_router.wal import _read_journal

STREAMING_PEAK_LIMIT_BYTES = 4 * 1024 * 1024


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_yields_each_json_object_and_skips_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "j.jsonl", '{"a": 1}\n\n   \n{"a": 2}\n')
    assert list(_read_journal(path)) == [{"a": 1}, {"a": 2}]


def test_accepts_a_valid_final_line_without_trailing_newline(tmp_path: Path) -> None:
    path = _write(tmp_path / "j.jsonl", '{"a": 1}\n{"a": 2}')
    assert list(_read_journal(path)) == [{"a": 1}, {"a": 2}]


def test_discards_an_unterminated_truncated_final_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "j.jsonl", '{"a": 1}\n{"a": 2, "tru')
    assert list(_read_journal(path)) == [{"a": 1}]


def test_rejects_a_terminated_malformed_final_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "j.jsonl", '{"a": 1}\n{"a": 2, "tru\n')
    with pytest.raises(ValueError, match=r"WAL corrupt at line 2"):
        list(_read_journal(path))


def test_reports_the_one_based_line_number_of_a_corrupt_middle_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "j.jsonl", '{"a": 1}\n\nnot json\n{"a": 3}\n')
    with pytest.raises(ValueError, match=r"WAL corrupt at line 3"):
        list(_read_journal(path))


@pytest.mark.parametrize("bad_line", ["[1, 2]", '"text"', "7", "NaN"])
def test_rejects_non_object_and_non_finite_lines(tmp_path: Path, bad_line: str) -> None:
    path = _write(tmp_path / "j.jsonl", f'{{"a": 1}}\n{bad_line}\n{{"a": 3}}\n')
    with pytest.raises(ValueError, match=r"WAL corrupt at line 2"):
        list(_read_journal(path))


def test_does_not_hold_the_whole_journal_in_memory(tmp_path: Path) -> None:
    line = '{"phase": "select", "pad": "' + "x" * 200 + '"}\n'
    path = _write(tmp_path / "big.jsonl", line * 60_000)  # ~13 MB on disk
    tracemalloc.start()
    try:
        count = sum(1 for _ in _read_journal(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 60_000
    assert peak < STREAMING_PEAK_LIMIT_BYTES
