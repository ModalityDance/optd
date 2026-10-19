#!/usr/bin/env python3
"""Build a deterministic 8-math/8-code prompt schedule for multi-teacher OPD.

This script only converts prompts and test metadata. It never executes code.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEEPCODER_DATASET_REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
# Vendored verbatim from the DeepCoder release harness in
# agentica-project/rllm:rllm/system_prompts.py.  Keep these strings stable so
# the supplemental experiment presents the same task contract as DeepCoder.
LCB_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)
LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
    "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def format_deepcoder_question(problem, starter_code=None):
    """Match DeepCoder's released ``fetch_live_code_bench_system_prompt``."""
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + str(problem)
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt

CODE_VALIDATION_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field(
            "prompt",
            pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])),
        ),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct([pa.field("style", pa.string()), pa.field("ground_truth", pa.string())]),
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("source", pa.string()),
                    pa.field("source_index", pa.string()),
                    pa.field("source_path", pa.string()),
                    pa.field("dataset_revision", pa.string()),
                ]
            ),
        ),
    ]
)


def _json_value(value):
    while isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def normalize_tests(value, fn_name=None):
    """Normalize common code-dataset test schemas to SandboxFusion inputs/outputs."""
    value = _json_value(value)
    if isinstance(value, dict):
        if "inputs" in value and "outputs" in value:
            inputs, outputs = value["inputs"], value["outputs"]
        elif "input" in value and "output" in value:
            inputs, outputs = [value["input"]], [value["output"]]
        elif "test" in value:
            return normalize_tests(value["test"], fn_name=fn_name)
        else:
            raise ValueError(f"unsupported test dictionary keys: {sorted(value)}")
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        inputs = [item["input"] for item in value if "input" in item and "output" in item]
        outputs = [item["output"] for item in value if "input" in item and "output" in item]
    else:
        raise ValueError(f"unsupported tests value: {type(value).__name__}")

    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]
    if not inputs or len(inputs) != len(outputs):
        raise ValueError(f"invalid tests: {len(inputs)} inputs vs {len(outputs)} outputs")
    tests = {"inputs": inputs, "outputs": outputs}
    if fn_name:
        tests["fn_name"] = str(fn_name)
    return tests


def _first_present(row, names):
    for name in names:
        if name in row and row[name] is not None:
            value = row[name]
            if not (isinstance(value, float) and np.isnan(value)):
                return value
    raise KeyError(f"none of the columns are present: {names}")


def convert_code_frame(frame, source_name, source_path):
    rows = []
    failures = 0
    for source_index, row in frame.iterrows():
        try:
            problem = str(_first_present(row, ("problem", "prompt", "question", "description")))
            metadata = _json_value(row.get("metadata", {}))
            metadata = metadata if isinstance(metadata, dict) else {}
            fn_name = metadata.get("func_name") or metadata.get("fn_name")
            tests = normalize_tests(
                _first_present(row, ("tests", "test_cases", "input_output", "test", "public_tests")),
                fn_name=fn_name,
            )
        except (KeyError, TypeError, ValueError):
            failures += 1
            continue
        starter_code = row.get("starter_code", "")
        starter_code = "" if starter_code is None else str(starter_code)
        prompt = format_deepcoder_question(problem, starter_code if starter_code.strip() else None)
        rows.append(
            {
                "data_source": "codecontests",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "CODE",
                "reward_model": {"style": "sandbox_fusion", "ground_truth": json.dumps(tests)},
                "extra_info": {
                    "source": source_name,
                    "source_index": str(source_index),
                    "source_path": str(source_path),
                    "dataset_revision": DEEPCODER_DATASET_REVISION,
                },
            }
        )
    if failures:
        print(f"{source_name}: skipped {failures} rows with unsupported/missing tests")
    return pd.DataFrame(rows)


def sample_frame(frame, count, seed):
    if len(frame) < count:
        raise ValueError(f"need {count} usable rows, found {len(frame)}")
    return frame.sample(n=count, replace=False, random_state=seed).reset_index(drop=True)


def write_code_validation_frame(frame, output_path):
    """Write string, not large_string, so it concatenates with existing verl val sets."""
    table = pa.Table.from_pylist(frame.to_dict("records"), schema=CODE_VALIDATION_SCHEMA)
    pq.write_table(table, output_path)


def allocate_code_quotas(code_frames, total):
    """Allocate near-even quotas, redistributing when a source shard is small."""
    source_count = len(code_frames)
    base_quota, remainder = divmod(total, source_count)
    quotas = [
        min(len(frame), base_quota + (1 if index < remainder else 0))
        for index, (_, frame) in enumerate(code_frames)
    ]
    missing = total - sum(quotas)
    while missing:
        progressed = False
        for index, (_, frame) in enumerate(code_frames):
            if quotas[index] < len(frame):
                quotas[index] += 1
                missing -= 1
                progressed = True
                if not missing:
                    break
        if not progressed:
            raise ValueError(f"need {total} usable code rows, found {sum(len(frame) for _, frame in code_frames)}")
    return quotas


def build_schedule(math_frame, code_frames, steps, seed):
    prompts_per_domain = steps * 8
    math = sample_frame(math_frame, prompts_per_domain, seed)

    code_parts = []
    quotas = allocate_code_quotas(code_frames, prompts_per_domain)
    for index, ((source_name, frame), quota) in enumerate(zip(code_frames, quotas, strict=True)):
        print(f"{source_name}: sampling {quota} rows")
        code_parts.append(sample_frame(frame, quota, seed + index + 1))
    code = pd.concat(code_parts, ignore_index=True).sample(frac=1, random_state=seed + 100).reset_index(drop=True)

    ordered = []
    for step in range(steps):
        math_block = math.iloc[step * 8 : (step + 1) * 8]
        code_block = code.iloc[step * 8 : (step + 1) * 8]
        for offset in range(8):
            ordered.append(math_block.iloc[offset].to_dict())
            ordered.append(code_block.iloc[offset].to_dict())
    return pd.DataFrame(ordered)


def build_full_pool(math_frame, code_frames, seed):
    """Keep every usable prompt; balancing belongs to the training sampler."""
    return pd.concat([math_frame, *(frame for _, frame in code_frames)], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", type=Path, default=Path("datasets/dapo-math-17k.parquet"))
    parser.add_argument("--code", type=Path, action="append", default=[])
    parser.add_argument("--code-source", action="append", default=[])
    parser.add_argument("--code-root", type=Path, help="DeepCoder revision root; load every taco/primeintellect/lcbv5 train shard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-code", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--pool-mode", choices=("fixed_2400", "full_pool"), default="fixed_2400")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.code_root is not None:
        if args.code or args.code_source:
            raise ValueError("--code-root cannot be combined with --code/--code-source")
        for source, expected in (("taco", 4), ("primeintellect", 5), ("lcbv5", 11)):
            paths = sorted((args.code_root / source).glob("train-*.parquet"))
            if len(paths) != expected:
                raise ValueError(f"{source}: expected {expected} train shards, found {len(paths)}")
            args.code.extend(paths)
            args.code_source.extend([source] * len(paths))
    if len(args.code) != len(args.code_source):
        raise ValueError("--code and --code-source must appear the same number of times")
    if not args.code:
        raise ValueError("at least one code parquet is required")

    math = pd.read_parquet(args.math)
    code_frames = []
    for source_name, path in zip(args.code_source, args.code, strict=True):
        converted = convert_code_frame(pd.read_parquet(path), source_name, path)
        print(f"{source_name}: {len(converted)} usable rows")
        code_frames.append((source_name, converted))

    schedule = (build_full_pool(math, code_frames, args.seed) if args.pool_mode == "full_pool"
                else build_schedule(math, code_frames, steps=args.steps, seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(args.output, index=False)
    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()

    counts = schedule["data_source"].value_counts().to_dict()
    code_source_counts = (
        schedule.loc[schedule["data_source"] == "codecontests", "extra_info"]
        .map(lambda info: info["source"])
        .value_counts()
        .to_dict()
    )
    manifest = {
        "output": str(args.output),
        "output_sha256": output_sha256,
        "rows": len(schedule),
        "pool_mode": args.pool_mode,
        "steps": args.steps,
        "batch_pattern": ("sampler controlled" if args.pool_mode == "full_pool"
                          else "M,C repeated eight times per 16-prompt outer batch"),
        "seed": args.seed,
        "data_source_counts": counts,
        "code_source_counts": code_source_counts,
        "math_path": str(args.math),
        "code_paths": [str(path) for path in args.code],
        "code_sources": args.code_source,
        "deepcoder_dataset_revision": DEEPCODER_DATASET_REVISION,
        "harness": "agentica-project/rllm DeepCoder single-turn prompt contract",
        "reward_semantics": "all_tests_pass_binary_validation_only",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    if (args.validation_code is None) != (args.validation_output is None):
        raise ValueError("--validation-code and --validation-output must be provided together")
    if args.validation_code is not None:
        validation = convert_code_frame(
            pd.read_parquet(args.validation_code), "codeforces_eval", args.validation_code
        )
        validation = sample_frame(validation, args.validation_count, args.seed + 1000)
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        write_code_validation_frame(validation, args.validation_output)
        print(f"wrote {len(validation)} code validation rows to {args.validation_output}")


if __name__ == "__main__":
    main()
