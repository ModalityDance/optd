# Training reference

Download students and teachers to local model directories and prepare datasets as described in [data preparation](data.md). Model weights, benchmark data, and trained checkpoints are not bundled. Preserve teacher vocabulary/tokenizer compatibility; the launcher runs model/data checks before allocating GPUs.

```bash
export ACTOR_MODEL_PATH=/path/to/student
export REWARD_MODEL_PATH=/path/to/math_teacher
export SECONDARY_REWARD_MODEL_PATH=/path/to/code_teacher
export TRAIN_DATASET=/path/to/prepared_train.parquet
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code

# Inspect settings without starting training or allocating GPUs.
python scripts/train.py --config configs/4b_late.json --method optd --dry-run

# Matched final-policy OPD control.
python scripts/train.py --config configs/4b_late.json --method opd

# OPTD: same configuration plus transition supervision.
python scripts/train.py --config configs/4b_late.json --method optd
```

Each invocation is a separate job. Use free GPUs, a distinct run name/output directory, and a running SandboxFusion service before launching training. `RUN_NAME` controls checkpoint and log names; an explicit environment variable overrides the selected JSON profile. Run `--smoke diagnostic` for a short pipeline check or `--smoke startup` for the inherited startup gate. Smoke jobs use console logging; formal jobs use console and W&B. Diagnostic smoke reduces lengths and does not reproduce the paper budget.

| Configuration | Intermediate blocks | Updates | Prompts/update | Prompts/minibatch |
| --- | --- | ---: | ---: | ---: |
| `1p5b_mid.json` | 7 / 13 / 19 | 150 | 16 | 8 |
| `1p5b_late.json` | 11 / 15 / 19 | 150 | 16 | 8 |
| `4b_mid.json` | 7 / 13 / 19 | 50 | 128 | 64 |
| `4b_late.json` | 14 / 19 / 24 | 50 | 128 | 64 |
| `7b_late.json` | 13 / 17 / 21 | 100 | 128 | 64 |
| `1p5b_three_domain.json` | 11 / 15 / 19 | 150 | 16 | 8 |

Block indices are zero-based; the final readout is appended automatically. Every prompt generates two responses. Qwen3-4B uses native non-thinking mode. All profiles use BF16, top-16 final-policy candidates, a learning rate of 1e-5 with 3% warmup, and transition supervision on the last 2,000 response positions. Dual-domain response limits are 16,384 tokens; three-domain limits are 10,240. These profiles transcribe the paper protocol; they do not bundle the original training artifacts or guarantee identical numerical results on a different software/hardware stack.

The three-domain configuration additionally requires `TERTIARY_REWARD_MODEL_PATH` and the Rubrics-To-Tokens dataset. `--method layeropd` switches to absolute-layer supervision with per-layer student top-16 support. The paper uses `1p5b_mid.json` and `4b_late.json` for its 1.5B/4B LayerOPD baselines; its 7B LayerOPD baseline uses blocks 11/15/19 (set both `LAYERWISE_OPD_STUDENT_LAYERS` and `LAYERWISE_OPD_TEACHER_LAYERS` to that path). OPRD/LastOPD and historical research launchers are outside the public entry point.


## Code and tests

```text
configs/                     Paper training profiles
scripts/train.py             Public OPD / OPTD / LayerOPD entry point
scripts/                     Data preparation and model/data checks
scripts/val/eval/            Generation, grading, and merge tools
verl/verl/utils/jlens_distillation.py   Transition objective and readout helpers
verl/verl/workers/actor/dp_actor.py      Student loss integration
verl/verl/workers/fsdp_workers.py       Teacher prediction collection
verl/verl/utils/dataset/               Domain-balanced sampling
tests/                     Numerical objective tests
docs/                      Method, data, evaluation, and project page
```

The internal name `jlens_distillation.py` is retained because training imports it directly. OPTD uses the raw depth-transition branch and does not require fitting a Jacobian lens.

## Validation

```bash
PYTHONPATH="$PWD/verl" python -m pytest -q tests
```

The tests cover transition-loss behavior, candidate readouts/support, and the exact reverse transition-KL gradient contract. They do not constitute a full distributed training reproduction.

