"""Shared loaders for the released DeepCoder LCB-v5 evaluation contract."""

import glob
import importlib.util
import json
from pathlib import Path

import pandas as pd
import os

REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDER_PATH = REPO_ROOT / "scripts" / "build_multiteacher_math_code_data.py"
_BUILDER_SPEC = importlib.util.spec_from_file_location("oprd_multiteacher_data_builder", _BUILDER_PATH)
if _BUILDER_SPEC is None or _BUILDER_SPEC.loader is None:
    raise ImportError(f"cannot load {_BUILDER_PATH}")
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)
format_deepcoder_question = _BUILDER.format_deepcoder_question
normalize_tests = _BUILDER.normalize_tests


def load_lcb_rows(data_glob: str) -> list[dict]:
    paths = [Path(path) for path in sorted(glob.glob(data_glob))]
    if not paths:
        raise FileNotFoundError(f"no parquet shards matched {data_glob!r}")

    rows = []
    for path in paths:
        frame = pd.read_parquet(path)
        for row_index, row in frame.iterrows():
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = metadata if isinstance(metadata, dict) else {}
            starter_code = row.get("starter_code", "")
            starter_code = "" if starter_code is None else str(starter_code)
            rows.append(
                {
                    "example_id": f"{path.name}:{row_index}",
                    "question": format_deepcoder_question(
                        row["problem"], starter_code if starter_code.strip() else None
                    ),
                    "tests": normalize_tests(
                        row["tests"], fn_name=metadata.get("func_name") or metadata.get("fn_name")
                    ),
                    "source_path": str(path),
                    "source_index": int(row_index),
                }
            )
    return rows


def discover_lcb_glob() -> str:
    """Find the pinned LCB-v5 test shards across known node mount layouts.

    The same CPFS dataset has appeared as ``lcb_test_full/extracted`` on
    Huawei nodes and as the revision-pinned ``.../lcbv5`` directory on A800.
    Resolve the path at runtime instead of making callers guess the mount.
    """
    candidates = [
        "/mnt/data/opd1/lcb_test_full/extracted/test-*.parquet",
        "/mnt/data/opd1/lcb_test_full/extracted/test-*.parquet",
        os.path.expanduser("~/data/datasets/DeepCoder-Preview-Dataset-177913a7bd43791646ef6a43645caa3c871ab3db/lcbv5/test-*.parquet"),
    ]
    failures = []
    for pattern in candidates:
        paths = sorted(glob.glob(pattern))
        if not paths:
            continue
        if len(paths) != 8:
            failures.append(f"{pattern}: found {len(paths)} shards, expected 8")
            continue
        try:
            rows = load_lcb_rows(pattern)
        except Exception as exc:
            failures.append(f"{pattern}: {type(exc).__name__}: {exc}")
            continue
        if len(rows) != 279:
            failures.append(f"{pattern}: found {len(rows)} problems, expected 279")
            continue
        return pattern
    detail = "\n".join(failures)
    raise FileNotFoundError(
        "Could not find a valid pinned LCB-v5 dataset (8 shards, 279 problems). "
        "Checked known Huawei and A800 mount layouts."
        + (f"\n{detail}" if detail else "")
    )


def format_rllm_deepcoder_chat_prompt(tokenizer, question: str, *, non_thinking=False) -> str:
    """Match rLLM's ``DeepseekQwenChatTemplateParser`` byte-for-byte.

    DeepCoder ships a ``LlamaTokenizerFast``.  rLLM therefore deliberately
    does not use the tokenizer's native chat template: it adds an EOS before
    the assistant marker and seeds the reasoning response with ``<think>``.
    """
    if non_thinking:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if not prompt.rstrip().endswith("<think>\n\n</think>"):
            raise ValueError("Expected Qwen3 closed empty thinking block")
        return prompt
    # Qwen3 checkpoints serialize the BOS id as pad_token rather than setting
    # bos_token explicitly; preserve the same byte-level prompt contract.
    bos_token = tokenizer.bos_token or tokenizer.pad_token
    if not bos_token or not tokenizer.eos_token:
        raise ValueError("DeepCoder tokenizer must define BOS (or pad) and EOS tokens")
    return (
        bos_token
        + "<｜User｜>"
        + question
        + tokenizer.eos_token
        + "<｜Assistant｜><think>\n"
    )
