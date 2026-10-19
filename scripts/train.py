#!/usr/bin/env python3
"""Launch a paper configuration with the original OPD/OPTD training driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", choices=("opd", "optd", "layeropd"), default="optd")
    parser.add_argument("--smoke", choices=("startup", "diagnostic"))
    parser.add_argument("--dry-run", action="store_true", help="print effective settings without allocating GPUs")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    profile = json.loads(config.read_text())
    if not isinstance(profile, dict) or not all(isinstance(k, str) and k.isupper() for k in profile):
        parser.error("configuration must be an object of uppercase environment keys")
    env = os.environ.copy()
    # Explicit environment settings override the selected paper profile.
    for key, value in profile.items():
        env.setdefault(key, str(value))
    if args.method == "layeropd":
        env["LAYERWISE_OPD_OBJECTIVE"] = "absolute"
        env["LAYERWISE_OPD_SUPPORT"] = "self_student"
    if args.smoke:
        env["STARTUP_SMOKE" if args.smoke == "startup" else "DIAGNOSTIC_SMOKE"] = "True"
        env["TRAINER_LOGGERS"] = "[console]"
    env.setdefault("PROJECT_CONDA_ENV", "optd")
    env.setdefault("ATTN_IMPLEMENTATION", "sdpa")
    env.setdefault("RUN_NAME", f"{config.stem}-{args.method}")
    env.setdefault("SANDBOX_FUSION_URL", "http://127.0.0.1:8080/run_code")
    env.setdefault("PYTHONPATH", str(ROOT / "verl"))
    mode = "control" if args.method == "opd" else "raw_layerwise"
    command = ["bash", str(ROOT / "scripts/launch_multiteacher_math_code_common.sh"), mode]
    keys = set(profile) | {"PROJECT_CONDA_ENV", "ATTN_IMPLEMENTATION", "RUN_NAME", "SANDBOX_FUSION_URL",
        "ACTOR_MODEL_PATH", "REWARD_MODEL_PATH", "SECONDARY_REWARD_MODEL_PATH", "TERTIARY_REWARD_MODEL_PATH",
        "TRAIN_DATASET", "CUDA_VISIBLE_DEVICES", "STARTUP_SMOKE", "DIAGNOSTIC_SMOKE"}
    if args.dry_run:
        print(json.dumps({"command": command, "settings": {k: env[k] for k in sorted(keys) if k in env}}, indent=2))
        return
    required = ["ACTOR_MODEL_PATH", "REWARD_MODEL_PATH", "SECONDARY_REWARD_MODEL_PATH", "TRAIN_DATASET"]
    if config.stem.endswith("three_domain"):
        required.append("TERTIARY_REWARD_MODEL_PATH")
    missing = [k for k in required if not env.get(k) or not Path(env[k]).exists()]
    if missing:
        parser.error("set existing local artifact paths for: " + ", ".join(missing))
    if "CUDA_VISIBLE_DEVICES" not in env:
        parser.error("set CUDA_VISIBLE_DEVICES to four free GPUs (7B also supports eight)")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)

if __name__ == "__main__":
    main()
