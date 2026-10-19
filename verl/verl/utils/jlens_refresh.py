# Copyright 2026 OPRD
"""Periodic student JLens refresh helpers (idea.md §1).

Re-fit the student Jacobian lens from an exported HF checkpoint on a fixed
probe corpus, then hot-swap the detached ``lens.pt`` into training workers.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def hf_checkpoint_has_weights(hf_dir: str | Path) -> bool:
    """True if ``huggingface/`` contains model weights (not tokenizer-only)."""
    path = Path(hf_dir)
    if not path.is_dir():
        return False
    patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).exists() for name in patterns) or any(path.glob("*.safetensors"))


def should_refresh_student_jlens(*, global_step: int, interval: int, is_last_step: bool = False) -> bool:
    """Refresh after completed steps ``interval, 2*interval, ...`` (not the final step)."""
    if is_last_step or interval <= 0 or global_step <= 0:
        return False
    return global_step % interval == 0


def run_student_jlens_refresh(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    probe_data: str | Path,
    response_token_data: str | Path | None = None,
    source_layers: str,
    text_key: str = "prompt",
    num_prompts: int = 100,
    seed: int = 42,
    max_seq_len: int = 128,
    max_response_positions: int = 2000,
    skip_first: int = 16,
    dim_batch: int = 64,
    dtype: str = "bfloat16",
    cuda_visible_devices: str,
    project_root: str | Path | None = None,
    python_executable: str | None = None,
) -> Path:
    """Fit a student JLens via ``scripts/jlens/fit_jlens.py`` on a selected GPU.

    The selected physical GPU may be an otherwise active training GPU: refresh
    is called synchronously after checkpointing, and a small 1.5B fit is able
    to coexist with the resident actor/rollout workers when sufficient headroom
    is provisioned.  Callers remain responsible for selecting a safe device.

    Returns the output directory containing ``lens.pt``.
    """
    if not str(cuda_visible_devices).strip():
        raise ValueError(
            "jalign_refresh requires a free GPU via jalign_refresh_cuda_visible_devices "
            "(or env JALIGN_REFRESH_CUDA_VISIBLE_DEVICES)"
        )
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    fit_script = root / "scripts" / "jlens" / "fit_jlens.py"
    if not fit_script.is_file():
        raise FileNotFoundError(f"fit_jlens.py not found: {fit_script}")

    model_path = Path(model_path).expanduser().resolve()
    probe_data = Path(probe_data).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hf_checkpoint_has_weights(model_path):
        raise FileNotFoundError(
            f"HF checkpoint lacks weights for JLens refresh: {model_path}. "
            "Enable actor checkpoint save_contents including 'hf_model'."
        )
    if not probe_data.exists():
        raise FileNotFoundError(f"JLens probe corpus not found: {probe_data}")
    if response_token_data is not None:
        response_token_data = Path(response_token_data).expanduser().resolve()
        if not response_token_data.exists():
            raise FileNotFoundError(f"response token corpus not found: {response_token_data}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    # Prefer the caller's interpreter (verl env); fall back to current.
    python_bin = python_executable or os.environ.get("JALIGN_REFRESH_PYTHON") or "python3"

    cmd = [
        python_bin,
        str(fit_script),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--source-layers",
        source_layers,
        "--num-prompts",
        str(int(num_prompts)),
        "--seed",
        str(int(seed)),
        "--dim-batch",
        str(int(dim_batch)),
        "--dtype",
        dtype,
        "--device",
        "cuda:0",
        "--no-resume",
        "--local-files-only",
    ]
    if response_token_data is None:
        cmd.extend(["--probe-data", str(probe_data), "--text-key", text_key,
                    "--max-seq-len", str(int(max_seq_len)), "--skip-first", str(int(skip_first))])
    else:
        cmd.extend(["--response-token-data", str(response_token_data),
                    "--max-response-positions", str(int(max_response_positions))])
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"student JLens refresh failed (exit={completed.returncode}, "
            f"elapsed={elapsed:.1f}s, output_dir={output_dir})"
        )
    lens_path = output_dir / "lens.pt"
    if not lens_path.is_file():
        raise FileNotFoundError(f"refresh finished but lens.pt missing: {lens_path}")
    return output_dir
