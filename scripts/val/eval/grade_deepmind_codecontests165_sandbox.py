#!/usr/bin/env python3
"""Fail-fast, resumable DeepMind CodeContests165 grading through SandboxFusion."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

from deepmind_codecontests165_common import DATASET_REVISION, load_deepmind_codecontests165_rows


def extract_python(response: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else None


def main() -> None:
    from verl.utils.reward_score.sandbox_fusion import utils as sandbox_utils

    sandbox_utils.API_TIMEOUT = 180
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:8080/run_code")
    args = parser.parse_args()
    rows = {row["example_id"]: row for row in load_deepmind_codecontests165_rows(args.data)}
    generations = [json.loads(line) for line in args.generations.read_text().splitlines() if line.strip()]
    if {item["example_id"] for item in generations} != set(rows):
        raise ValueError("generation file does not exactly cover the DeepMind CodeContests165 test split")
    jobs = [(f"{item['example_id']}:{i}", item["example_id"], response)
            for item in generations for i, response in enumerate(item["responses"])]
    old = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                old[item["key"]] = item

    def grade(job: tuple[str, str, str]) -> dict:
        key, example_id, response = job
        digest = hashlib.sha256(response.encode()).hexdigest()
        if key in old:
            if old[key]["sha256"] != digest:
                raise ValueError(f"response changed for resumed job {key}")
            return old[key]
        code = extract_python(response)
        checked = 0
        score = 0
        if code is not None:
            score = 1
            tests = rows[example_id]["tests"]
            for i, (stdin, stdout) in enumerate(zip(tests["inputs"], tests["outputs"], strict=True)):
                result, meta = sandbox_utils._process_single_case(
                    i, stdin, stdout, args.sandbox_url, code, 10, 1024, "python", None, None, True)
                if result == -1:
                    raise RuntimeError(f"infrastructure error: {key}, case {i}: {meta.get('status')}")
                checked += 1
                if result is not True:
                    score = 0
                    break
        return {"key": key, "example_id": example_id, "sha256": digest,
                "score": score, "tests_checked": checked, "code_extracted": code is not None}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = dict(old)
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(grade, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            if item["key"] not in records:
                handle.write(json.dumps(item) + "\n")
                handle.flush()
                records[item["key"]] = item
    grouped = {pid: [records[key]["score"] for key, job_pid, _ in jobs if job_pid == pid]
               for pid in rows}
    candidates = sum(len(scores) for scores in grouped.values())
    passed = sum(map(sum, grouped.values()))
    solved = sum(any(scores) for scores in grouped.values())
    n_values = {len(scores) for scores in grouped.values()}
    n = n_values.pop() if len(n_values) == 1 else None
    summary = {
        "benchmark": "deepmind_codecontests165", "upstream_dataset": "deepmind/code_contests",
        "split": "test",
        "dataset_revision": DATASET_REVISION, "problems": len(rows),
        "candidates": candidates, "candidates_per_problem": n,
        "passed_candidates": passed, "solved_problems": solved,
        "mean_at_n_percent": 100 * passed / candidates,
        "pass_at_n_percent": 100 * solved / len(rows),
        "test_groups": ["public_tests", "private_tests", "generated_tests"],
        "reward_semantics": "all_tests_pass_binary", "language": "python3",
        "timeout_seconds_per_case": 10, "memory_limit_mb": 1024,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
