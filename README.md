<div align="center">

# OPTD: On-Policy Transition Distillation

**Learning how a teacher refines its predictions across depth.**

[Overview](#overview) · [Getting started](#getting-started) · [Training guide](docs/training.md) · [Evaluation](docs/evaluation.md)

</div>

Code for **On-Policy Transition Distillation (OPTD)**, a method for transferring specialist capabilities into a single language model.

## Overview

As a language model processes a prompt, its predictions change from layer to layer. These changes reveal how the model gradually builds its prediction. Can a student learn from this process as well as from the teacher's final prediction?

OPTD brings this additional supervision to **on-policy distillation**, where the student generates responses and learns from a teacher's predictions on those same responses. Alongside matching the teacher's final prediction, OPTD trains the student to match **how the teacher changes its predictions across layers**.

![Overview of On-Policy Transition Distillation](assets/overview.png)

For a given response prefix, we compare how teacher and student change the log probability of the same candidate token between two selected layers. This gives the student a training signal about the teacher's prediction changes along that part of the network. We apply this supervision across selected layer segments while retaining the standard final-output distillation objective.

We study OPTD with DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3-4B, using specialists in mathematics, coding, and instruction following. Each training prompt is supervised by the specialist for its domain. See [method details](docs/method.md) for the objective and implementation.

## Getting started

### 1. Install

Use Linux with NVIDIA GPUs, a CUDA-12-compatible driver, and Python 3.12:

```bash
conda create -n optd python=3.12 -y
conda activate optd
bash scripts/setup_oprd_env.sh
```

### 2. Prepare models and data

Download a student and its domain teachers, then prepare the training data using the [data preparation guide](docs/data.md). The guide lists the models and datasets used in the paper.

Set the local paths and available GPUs. Coding tasks also require a running SandboxFusion service:

```bash
export ACTOR_MODEL_PATH=/path/to/student
export REWARD_MODEL_PATH=/path/to/math_teacher
export SECONDARY_REWARD_MODEL_PATH=/path/to/code_teacher
export TRAIN_DATASET=/path/to/prepared_train.parquet
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code
```

### 3. Train

For example, train Qwen3-4B with the paper's Late configuration:

```bash
python scripts/train.py --config configs/4b_late.json --method optd
```

To run the OPD baseline with the same configuration, use `--method opd`. Add `--dry-run` to inspect the settings before launching a job.

The [training guide](docs/training.md) covers the other model sizes, the three-domain setting, configuration options, and tests. Models, datasets, and trained checkpoints are separate from this source release.

## Evaluation

We evaluate mathematics on AIME24/25 and AIMO; coding on LiveCodeBench v5, CodeContests, and Codeforces; and instruction following on IFEval.

See the [evaluation guide](docs/evaluation.md) for benchmark protocols, generation and grading commands, and required external resources.

## Acknowledgements and license

This implementation builds on OPD and verl. See [third-party notices](THIRD_PARTY_NOTICES.md), the [repository license](LICENSE), and the [verl license](verl/LICENSE).
