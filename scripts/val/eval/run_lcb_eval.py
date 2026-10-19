#!/usr/bin/env python3
"""Run the standard LCB generation, SandboxFusion grading, and result checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="merged HuggingFace model; optional if generations are complete")
    parser.add_argument("--data-glob", help="LCB parquet glob; omitted means discover and validate it")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--generations", type=Path, help="reuse an existing generation file")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:8080/run_code")
    parser.add_argument("--non-thinking", action="store_true")
    args = parser.parse_args()

    from deepcoder_lcb_common import load_lcb_rows
    from deepcoder_lcb_common import discover_lcb_glob

    if args.data_glob is None:
        args.data_glob = discover_lcb_glob()
        print(f"Discovered validated LCB dataset: {args.data_glob}", flush=True)

    here = Path(__file__).resolve().parent
    generations = args.generations or Path(str(args.output_prefix) + ".jsonl")
    graded = Path(str(args.output_prefix) + ".graded.jsonl")
    summary = graded.with_suffix(".summary.json")
    rows = load_lcb_rows(args.data_glob)
    expected = {row["example_id"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("dataset contains duplicate example_id values")

    def read_generations() -> dict:
        if not generations.exists():
            return {}
        records = {}
        with generations.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item["example_id"]
                if key in records or key not in expected:
                    raise ValueError(f"duplicate or unexpected example_id: {key}")
                if len(item.get("responses", [])) != 4:
                    raise ValueError(f"example {key} does not contain four responses")
                records[key] = item
        return records

    existing = read_generations()
    if len(existing) != len(expected):
        if not args.model_path:
            parser.error(f"{len(existing)}/{len(expected)} generations present; --model-path is required")
        if args.tensor_parallel_size < 1:
            parser.error("--tensor-parallel-size must be positive")
        print(f"Generating {len(expected) - len(existing)} missing LCB problems", flush=True)
        subprocess.run([
            sys.executable, str(here / "eval_deepcoder_lcb_vllm.py"),
            *( ["--non-thinking"] if args.non_thinking else [] ),
            "--model-path", args.model_path, "--data-glob", args.data_glob,
            "--output", str(generations), "--tensor-parallel-size", str(args.tensor_parallel_size),
        ], check=True)
    else:
        print(f"Reusing all {len(expected)} LCB generation records", flush=True)
    if set(read_generations()) != expected:
        raise ValueError("generation file does not cover the complete LCB dataset")

    if args.workers < 1:
        parser.error("--workers must be positive")
    ping_url = args.sandbox_url.rsplit("/", 1)[0] + "/v1/ping"
    with urlopen(ping_url, timeout=10) as response:
        if response.read().decode().strip().strip('"') != "pong":
            raise RuntimeError(f"SandboxFusion ping failed: {ping_url}")
    env = os.environ.copy()
    repo = here.parents[2]
    env["PYTHONPATH"] = str(repo / "verl") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Grading {len(expected) * 4} candidates through {args.sandbox_url}", flush=True)
    subprocess.run([
        sys.executable, str(here / "grade_lcb_failfast.py"),
        "--data-glob", args.data_glob, "--generations", str(generations),
        "--output", str(graded), "--workers", str(args.workers),
        "--sandbox-url", args.sandbox_url,
    ], check=True, env=env)
    result = json.loads(summary.read_text())
    if result["problems"] != len(expected) or result["candidates"] != len(expected) * 4:
        raise ValueError(f"incomplete LCB summary: {result}")
    print(f"LCB complete: mean@4={result['mean_at_4_percent']:.4f}%, "
          f"pass@4={result['pass_at_4_percent']:.4f}%; {summary}", flush=True)


if __name__ == "__main__":
    main()
