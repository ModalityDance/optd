#!/usr/bin/env python3
"""Fail-fast validation-dispatch and import-path checks for multi-teacher runs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
from unittest.mock import patch


MATH_SOURCES = ("AMC23", "AIME24", "AIME25", "AIMO")
EXPECTED_SOURCES = set(MATH_SOURCES) | {"codecontests"}
REQUIRED_DATASET_SOURCES = EXPECTED_SOURCES


def _module_report(expected_root: str) -> dict[str, str]:
    import verl
    import verl.utils.reward_score as reward_score
    import verl.workers.reward_manager.naive as naive

    expected_verl = (Path(expected_root).resolve() / "verl").resolve()
    paths = {
        "verl": str(Path(inspect.getfile(verl)).resolve()),
        "reward_score": str(Path(inspect.getfile(reward_score)).resolve()),
        "naive": str(Path(inspect.getfile(naive)).resolve()),
    }
    for name, module_path in paths.items():
        if not Path(module_path).is_relative_to(expected_verl):
            raise RuntimeError(
                f"{name} imported from {module_path}, expected a module below {expected_verl}"
            )

    reward_path = Path(paths["reward_score"])
    paths["reward_score_sha256"] = hashlib.sha256(reward_path.read_bytes()).hexdigest()
    return paths


def _dispatch_smoke(expected_root: str) -> dict[str, object]:
    report: dict[str, object] = {"modules": _module_report(expected_root)}

    from verl.utils.reward_score import default_compute_score
    from verl.workers.reward_manager.naive import NaiveRewardManager

    signature = inspect.signature(NaiveRewardManager)
    required_kwargs = {
        "skip_rule_reward_when_rm_scores",
        "code_reward_continuous",
        "code_reward_all_passed",
        "code_reward_lcb_compat",
    }
    missing = required_kwargs - set(signature.parameters)
    if missing:
        raise RuntimeError(f"NaiveRewardManager is missing required kwargs: {sorted(missing)}")

    passed: list[str] = []
    for source in MATH_SOURCES:
        with patch("verl.utils.reward_score.math_dapo.compute_score", return_value=1.0):
            score = default_compute_score(source, "solution", "answer")
        if score != 1.0:
            raise RuntimeError(f"unexpected smoke score for {source}: {score!r}")
        passed.append(source)

    with patch("verl.utils.reward_score.sandbox_fusion.compute_score", return_value=(1.0, [])):
        score = default_compute_score(
            "codecontests",
            "```python\nprint(1)\n```",
            '{"inputs":[""],"outputs":["1"]}',
            sandbox_fusion_url="http://preflight.invalid/run_code",
            memory_limit_mb=1024,
            code_reward_continuous=False,
            code_reward_all_passed=True,
            code_reward_lcb_compat=True,
        )
    if score != 1.0:
        raise RuntimeError(f"unexpected smoke score for codecontests: {score!r}")
    passed.append("codecontests")
    report["dispatch_sources"] = passed
    return report


def _dataset_sources(paths: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    found: set[str] = set()
    for path in paths:
        table = pq.read_table(path, columns=["data_source"])
        values = {str(value) for value in table.column("data_source").to_pylist()}
        task = Path(path).parent.name
        if task in MATH_SOURCES and values != {task}:
            raise RuntimeError(f"{path}: expected data_source={task}, found {sorted(values)}")
        if not values:
            raise RuntimeError(f"validation parquet is empty: {path}")
        found.update(values)
    unexpected = found - EXPECTED_SOURCES
    missing = REQUIRED_DATASET_SOURCES - found
    if unexpected or missing:
        raise RuntimeError(
            f"validation data sources differ from the expected set; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, found={sorted(found)}"
        )
    return sorted(found)


def _sandbox_execution_smoke(sandbox_url: str) -> dict[str, object]:
    """Verify the exact POST endpoint used by the validation reward client."""
    from verl.utils.reward_score.sandbox_fusion.utils import call_sandbox_api

    response, error = call_sandbox_api(
        sandbox_url,
        "print(1)",
        "",
        compile_timeout=2,
        run_timeout=2,
        memory_limit_mb=256,
    )
    if error is not None or response is None:
        raise RuntimeError(f"sandbox execution smoke failed for {sandbox_url}: {error}")
    run_result = response.get("run_result", {})
    if response.get("status") != "Success" or run_result.get("return_code") != 0 or run_result.get("stdout") != "1\n":
        raise RuntimeError(f"sandbox execution smoke returned an unexpected response: {response!r}")
    return {"url": sandbox_url, "status": response["status"], "stdout": run_result["stdout"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--validation-file", action="append", default=[])
    parser.add_argument("--ray-address")
    parser.add_argument("--sandbox-url")
    args = parser.parse_args()

    result: dict[str, object] = {
        "driver": _dispatch_smoke(args.expected_root),
        "dataset_sources": _dataset_sources(args.validation_file),
    }
    if args.sandbox_url:
        result["sandbox_execution"] = _sandbox_execution_smoke(args.sandbox_url)

    if args.ray_address:
        import ray

        ray.init(address=args.ray_address, logging_level="ERROR")
        worker_smoke = ray.remote(num_cpus=1)(_dispatch_smoke)
        result["ray_worker"] = ray.get(worker_smoke.remote(args.expected_root))
        ray.shutdown()

    print("MULTITEACHER_PREFLIGHT_OK", result, flush=True)


if __name__ == "__main__":
    main()
