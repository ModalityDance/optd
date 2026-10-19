"""Pinned DeepMind CodeContests 165-problem test-set helpers."""

from __future__ import annotations

import hashlib
import fcntl
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from deepcoder_lcb_common import format_rllm_deepcoder_chat_prompt

DATASET_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
TEST_FILENAME = "test-00000-of-00001-9c49eeff30aacaa8.parquet"
TEST_URL = (
    "https://huggingface.co/datasets/deepmind/code_contests/resolve/"
    f"{DATASET_REVISION}/data/{TEST_FILENAME}"
)
SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a competitive programming "
    "problem. Write a correct Python 3 program that reads from standard input and writes "
    "to standard output. Enclose the complete program in a ```python code block."
)


def ensure_test_parquet(path: Path) -> Path:
    """Download the revision-pinned HF parquet atomically when absent."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return path
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.part")
        try:
            with urlopen(TEST_URL, timeout=60) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _values(group: object, field: str) -> list[str]:
    if not isinstance(group, dict):
        return []
    values = group.get(field, [])
    return [str(value) for value in list(values)]


def problem_id(row: pd.Series, index: int) -> str:
    if int(row.get("cf_contest_id", 0)) and str(row.get("cf_index", "")):
        return f"cf-{int(row['cf_contest_id'])}-{row['cf_index']}"
    digest = hashlib.sha256(f"{index}\0{row['name']}".encode()).hexdigest()[:16]
    return f"deepmind-codecontests165-{digest}"


def load_deepmind_codecontests165_rows(path: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    rows = []
    for index, row in frame.iterrows():
        inputs: list[str] = []
        outputs: list[str] = []
        counts = {}
        # This matches DeepMind's example executor: public, private, then generated.
        for group_name in ("public_tests", "private_tests", "generated_tests"):
            group_inputs = _values(row[group_name], "input")
            group_outputs = _values(row[group_name], "output")
            if len(group_inputs) != len(group_outputs):
                raise ValueError(f"row {index} has mismatched {group_name}")
            inputs.extend(group_inputs)
            outputs.extend(group_outputs)
            counts[group_name] = len(group_inputs)
        if not inputs:
            raise ValueError(f"row {index} has no tests")
        pid = problem_id(row, int(index))
        prompt = f"{SYSTEM_PROMPT}\n\n{row['description']}\n\n### Answer:\n"
        rows.append({
            "example_id": pid,
            "name": str(row["name"]),
            "question": prompt,
            "tests": {"inputs": inputs, "outputs": outputs},
            "test_counts": counts,
            "source": int(row["source"]),
            "difficulty": int(row["difficulty"]),
            "cf_rating": int(row["cf_rating"]),
            "source_index": int(index),
        })
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("DeepMind CodeContests165 test set has duplicate problem IDs")
    return rows


__all__ = [
    "DATASET_REVISION", "TEST_FILENAME", "TEST_URL", "ensure_test_parquet",
    "format_rllm_deepcoder_chat_prompt", "load_deepmind_codecontests165_rows",
]
