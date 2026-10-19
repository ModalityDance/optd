#!/usr/bin/env python3
"""Grade DeepCoder LCB-v5 generations with binary all-hidden-tests pass."""

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path

from deepcoder_lcb_common import discover_lcb_glob, load_lcb_rows
from verl.utils.reward_score.sandbox_fusion import compute_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sandbox-url", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    if args.data_glob is None:
        args.data_glob = discover_lcb_glob()
        print(f"Discovered validated LCB dataset: {args.data_glob}", flush=True)

    tests_by_id = {row["example_id"]: row["tests"] for row in load_lcb_rows(args.data_glob)}
    generations = [json.loads(line) for line in args.generations.read_text().splitlines() if line.strip()]
    if len(generations) != len(tests_by_id):
        raise ValueError(f"expected {len(tests_by_id)} generations, found {len(generations)}")
    if {item["example_id"] for item in generations} != set(tests_by_id):
        raise ValueError("generation IDs do not exactly match the LCB-v5 test split")

    semaphore = threading.Semaphore(args.workers)

    def grade_one(example_id: str, response: str) -> tuple[float, int]:
        score, metadata = compute_score(
            args.sandbox_url,
            semaphore,
            args.memory_limit_mb,
            response,
            tests_by_id[example_id],
            continuous=False,
            timeout=args.timeout,
            all_or_nothing=True,
            lcb_compat=True,
        )
        return score, len(metadata)

    def grade(item: dict) -> dict:
        # New files contain one problem with ``responses`` from vLLM n=4.
        # Accept the old singular ``response`` format for existing results.
        responses = item.get("responses")
        if responses is None:
            responses = [item["response"]]
        graded = [grade_one(item["example_id"], response) for response in responses]
        scores = [score for score, _ in graded]
        return {
            "example_id": item["example_id"],
            "scores": scores,
            "mean_score": sum(scores) / len(scores),
            "pass_at_n": int(any(scores)),
            "num_candidates": len(scores),
            "num_tests": [num_tests for _, num_tests in graded],
            "finish_reasons": item.get("finish_reasons", [item.get("finish_reason")]),
            "response_tokens": item.get("response_tokens"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        graded = list(pool.map(grade, generations))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in graded:
            handle.write(json.dumps(item) + "\n")
    mean_at_n = sum(item["mean_score"] for item in graded) / len(graded)
    pass_at_n = sum(item["pass_at_n"] for item in graded) / len(graded)
    n_values = {item["num_candidates"] for item in graded}
    n = n_values.pop() if len(n_values) == 1 else None
    summary = {
        "problems": len(graded),
        "candidates_per_problem": n,
        "mean_at_n": mean_at_n,
        "mean_at_n_percent": 100 * mean_at_n,
        "pass_at_n": pass_at_n,
        "pass_at_n_percent": 100 * pass_at_n,
        # Keep the old key for n=1 consumers.
        "passed": int(sum(item["pass_at_n"] for item in graded)),
        "pass_at_1": mean_at_n if n == 1 else None,
        "pass_at_1_percent": 100 * mean_at_n if n == 1 else None,
        "reward_semantics": "all_tests_pass_binary",
        "grader_harness": "rllm_lcb_compat_7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
