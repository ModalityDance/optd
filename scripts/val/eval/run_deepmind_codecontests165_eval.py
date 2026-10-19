#!/usr/bin/env python3
"""Run pinned DeepMind CodeContests generation, grading, and completeness checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

from deepmind_codecontests165_common import (
    TEST_FILENAME,
    ensure_test_parquet,
    load_deepmind_codecontests165_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="merged HuggingFace model; optional with complete generations")
    parser.add_argument("--data", type=Path,
                        default=Path("datasets/test_data/DeepMind-CodeContests165") / TEST_FILENAME)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:8080/run_code")
    parser.add_argument("--non-thinking", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    data = ensure_test_parquet(args.data)
    expected = {row["example_id"] for row in load_deepmind_codecontests165_rows(data)}
    generations = args.generations or Path(str(args.output_prefix) + ".jsonl")
    existing = []
    if generations.exists():
        existing = [json.loads(line) for line in generations.read_text().splitlines() if line.strip()]
    complete = ({item["example_id"] for item in existing} == expected and
                all(len(item.get("responses", [])) == args.n for item in existing))
    if not complete:
        if not args.model_path:
            parser.error("incomplete generations require --model-path")
        subprocess.run([
            sys.executable, str(here / "eval_deepmind_codecontests165_vllm.py"),
            *( ["--non-thinking"] if args.non_thinking else [] ),
            "--model-path", args.model_path, "--data", str(data), "--output", str(generations),
            "--tensor-parallel-size", str(args.tensor_parallel_size), "--n", str(args.n),
        ], check=True)
    ping_url = args.sandbox_url.rsplit("/", 1)[0] + "/v1/ping"
    with urlopen(ping_url, timeout=10) as response:
        if response.read().decode().strip().strip('"') != "pong":
            raise RuntimeError(f"SandboxFusion ping failed: {ping_url}")
    graded = Path(str(args.output_prefix) + ".graded.jsonl")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(here.parents[2] / "verl") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([
        sys.executable, str(here / "grade_deepmind_codecontests165_sandbox.py"),
        "--data", str(data), "--generations", str(generations), "--output", str(graded),
        "--workers", str(args.workers), "--sandbox-url", args.sandbox_url,
    ], check=True, env=env)
    summary = json.loads(graded.with_suffix(".summary.json").read_text())
    if summary["problems"] != len(expected) or summary["candidates"] != len(expected) * args.n:
        raise ValueError(f"incomplete DeepMind CodeContests165 summary: {summary}")
    print(f"DeepMind CodeContests165 complete: mean@{args.n}={summary['mean_at_n_percent']:.4f}%, "
          f"pass@{args.n}={summary['pass_at_n_percent']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
