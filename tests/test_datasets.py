"""Tests for dataset loading utilities."""

import json
import tempfile
from pathlib import Path

from tracellm.training.datasets import (
    load_code_folder,
    load_local_file,
    CODE_EXTENSIONS,
)


def test_load_code_folder(tmp_path):
    """Should load Python files from a folder."""
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def add(a, b): return a + b")
    (tmp_path / "readme.md").write_text("# Project")  # .md is in CODE_EXTENSIONS
    (tmp_path / "image.png").write_bytes(b"\x89PNG")    # should be skipped

    ds = load_code_folder(str(tmp_path))
    # .py + .md = 3 files, .png skipped
    assert len(ds) == 3


def test_load_code_folder_with_header(tmp_path):
    """Files should include the path header by default."""
    (tmp_path / "app.py").write_text("import os")
    ds = load_code_folder(str(tmp_path), add_file_header=True)
    assert "# File: app.py" in ds[0]["text"]


def test_load_code_folder_skips_hidden(tmp_path):
    """Hidden files and directories should be skipped."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("hidden")
    (tmp_path / ".env").write_text("SECRET=abc")  # no matching extension anyway
    (tmp_path / "main.py").write_text("code")

    ds = load_code_folder(str(tmp_path))
    assert len(ds) == 1


def test_load_code_folder_skips_large_files(tmp_path):
    """Files over MAX_CODE_FILE_BYTES should be skipped."""
    (tmp_path / "big.py").write_text("x" * (300 * 1024))  # 300 KB > 256 KB limit
    (tmp_path / "small.py").write_text("print('hi')")

    ds = load_code_folder(str(tmp_path))
    assert len(ds) == 1


def test_load_local_jsonl(tmp_path):
    """Should load a JSONL file."""
    data = [{"text": "line 1"}, {"text": "line 2"}, {"text": "line 3"}]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data))

    ds = load_local_file(str(path))
    assert len(ds) == 3
    assert ds[0]["text"] == "line 1"


def test_load_local_json(tmp_path):
    """Should load a JSON array file."""
    data = [{"text": "a"}, {"text": "b"}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))

    ds = load_local_file(str(path))
    assert len(ds) == 2


def test_load_local_txt(tmp_path):
    """Should load a plain text file, one line per example."""
    path = tmp_path / "corpus.txt"
    path.write_text("First line\nSecond line\nThird line\n")

    ds = load_local_file(str(path))
    assert len(ds) == 3
