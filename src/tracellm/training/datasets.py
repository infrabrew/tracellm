"""Dataset loading — HuggingFace Hub, local files, and code folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, concatenate_datasets

from tracellm.utils.logging import get_logger

log = get_logger("tracellm.training.datasets")

# Extensions recognized as code when loading from folders
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
    ".lua", ".r", ".jl", ".zig", ".v", ".nim", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".sql", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".css", ".scss",
    ".md", ".rst", ".txt", ".dockerfile", ".tf", ".hcl",
}

# Max file size to include (256 KB)
MAX_CODE_FILE_BYTES = 256 * 1024


def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
    text_field: str = "text",
    max_samples: int | None = None,
    streaming: bool = False,
) -> Dataset:
    """Load a dataset from HuggingFace Hub.

    Handles common dataset formats:
    - Single text field (e.g. "text")
    - Instruction format (instruction/input/output)
    - Chat format (messages list)
    """
    log.info(f"Loading HuggingFace dataset: {dataset_id} (split={split})")

    ds = load_dataset(dataset_id, split=split, streaming=streaming)

    if max_samples and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Check if the text_field exists
    if hasattr(ds, "column_names") and text_field in ds.column_names:
        return ds

    # Try to auto-detect and convert instruction format
    cols = set(ds.column_names) if hasattr(ds, "column_names") else set()

    if {"instruction", "output"}.issubset(cols):
        log.info("Detected instruction format — converting to text field")
        def format_instruction(example):
            inp = example.get("input", "")
            if inp:
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{example['output']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            return {"text": text}
        ds = ds.map(format_instruction, remove_columns=list(cols))
        return ds

    if "messages" in cols:
        log.info("Detected chat format — converting to text field")
        def format_chat(example):
            parts = []
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return {"text": "\n".join(parts)}
        ds = ds.map(format_chat, remove_columns=list(cols))
        return ds

    # If nothing matches, use the first string column
    if cols:
        first_col = list(cols)[0]
        log.warning(f"Field '{text_field}' not found, using '{first_col}' instead")
        ds = ds.rename_column(first_col, "text")
        return ds

    return ds


def load_code_folder(
    folder_path: str,
    recursive: bool = True,
    extensions: set[str] | None = None,
    add_file_header: bool = True,
) -> Dataset:
    """Load code files from a local folder as a training dataset.

    Each file becomes one training example with optional file path header.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Code folder not found: {folder}")

    exts = extensions or CODE_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    texts = []

    for file_path in sorted(folder.glob(pattern)):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue
        if file_path.stat().st_size > MAX_CODE_FILE_BYTES:
            log.debug(f"Skipping large file: {file_path}")
            continue
        # Skip hidden files and directories
        if any(part.startswith(".") for part in file_path.parts):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not content.strip():
            continue

        if add_file_header:
            rel_path = file_path.relative_to(folder)
            text = f"# File: {rel_path}\n\n{content}"
        else:
            text = content

        texts.append(text)

    if not texts:
        raise ValueError(f"No code files found in {folder} with extensions {exts}")

    log.info(f"Loaded {len(texts)} code files from {folder}")
    return Dataset.from_dict({"text": texts})


def load_multiple_code_folders(
    folder_paths: list[str],
    **kwargs,
) -> Dataset:
    """Load and concatenate code from multiple folders."""
    datasets = []
    for path in folder_paths:
        try:
            ds = load_code_folder(path, **kwargs)
            datasets.append(ds)
        except (FileNotFoundError, ValueError) as e:
            log.warning(f"Skipping folder {path}: {e}")

    if not datasets:
        raise ValueError("No valid code folders provided")

    return concatenate_datasets(datasets)


def load_local_file(file_path: str, text_field: str = "text") -> Dataset:
    """Load a local JSONL, CSV, or text file as a dataset."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    elif suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return Dataset.from_list(data)
        return Dataset.from_dict(data)

    elif suffix == ".csv":
        return load_dataset("csv", data_files=str(path), split="train")

    elif suffix == ".txt":
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        return Dataset.from_dict({"text": lines})

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def prepare_dataset(
    source: str,
    text_field: str = "text",
    split: str = "train",
    max_samples: int | None = None,
    code_paths: list[str] | None = None,
) -> Dataset:
    """Unified dataset loader — auto-detects source type.

    Args:
        source: HuggingFace dataset ID, local file path, or "code:" prefix.
        text_field: The text column name.
        split: Dataset split for HuggingFace datasets.
        max_samples: Limit samples.
        code_paths: Additional code folders to concatenate.
    """
    datasets_to_merge = []

    # Load primary source
    path = Path(source).expanduser()
    if source.startswith("code:"):
        folder = source[5:]
        datasets_to_merge.append(load_code_folder(folder))
    elif path.exists() and path.is_dir():
        datasets_to_merge.append(load_code_folder(str(path)))
    elif path.exists() and path.is_file():
        datasets_to_merge.append(load_local_file(str(path), text_field=text_field))
    else:
        # Assume HuggingFace dataset ID
        datasets_to_merge.append(
            load_hf_dataset(source, split=split, text_field=text_field, max_samples=max_samples)
        )

    # Add code paths
    if code_paths:
        for cp in code_paths:
            try:
                datasets_to_merge.append(load_code_folder(cp))
            except (FileNotFoundError, ValueError) as e:
                log.warning(f"Skipping code path {cp}: {e}")

    if len(datasets_to_merge) == 1:
        return datasets_to_merge[0]

    return concatenate_datasets(datasets_to_merge)
