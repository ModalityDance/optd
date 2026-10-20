<div align="center">
  <h1>OPTD: On-Policy Transition Distillation</h1>
  <a href="https://github.com/ModalityDance/optd">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&amp;logo=github" alt="GitHub Code">
  </a>
  <a href="https://huggingface.co/collections/ModalityDance/optd">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-fcc21b?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face Models">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
  </a>
  <p>
    <a href="#overview">Overview</a> ·
    <a href="#checkpoints">Models</a> ·
    <a href="docs/inference.md">Inference</a> ·
    <a href="#getting-started">Training</a> ·
    <a href="docs/evaluation.md">Evaluation</a>
  </p>
</div>

Code for **On-Policy Transition Distillation (OPTD)**, a method for transferring specialist capabilities into a single language model.

## Overview

As a language model processes a prompt, its predictions change from layer to layer. These changes reveal how the model gradually builds its prediction. Can a student learn from this process as well as from the teacher's final prediction?

OPTD brings this additional supervision to **on-policy distillation**, where the student generates responses and learns from a teacher's predictions on those same responses. Alongside matching the teacher's final prediction, OPTD trains the student to match **how the teacher changes its predictions across layers**.

![Overview of On-Policy Transition Distillation](assets/overview.png)

For a given response prefix, we compare how teacher and student change the log probability of the same candidate token between two selected layers. This gives the student a training signal about the teacher's prediction changes along that part of the network. We apply this supervision across selected layer segments while retaining the standard final-output distillation objective.

We study OPTD with DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3-4B, using specialists in mathematics, coding, and instruction following. Each training prompt is supervised by the specialist for its domain. See [method details](docs/method.md) for the objective and implementation.

## Checkpoints

Six trained OPTD models are available for inference and evaluation. Choose a model size and the training domains relevant to your task:

| Base model | Training domains | Available models | Generation mode |
| --- | --- | --- | --- |
| R1-Distill-Qwen-1.5B | Math + coding | [Mid](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Mid) · [Late](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Late) | Reasoning |
| R1-Distill-Qwen-1.5B | Math + coding + instruction following | [Three-domain](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-ThreeDomain) | Reasoning |
| Qwen3-4B | Math + coding | [Mid](https://huggingface.co/ModalityDance/OPTD-Qwen3-4B-Mid) · [Late](https://huggingface.co/ModalityDance/OPTD-Qwen3-4B-Late) | Non-thinking |
| R1-Distill-Qwen-7B | Math + coding | [Late](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-7B-Late) | Reasoning |

**Mid** and **Late** refer to where transition supervision is applied during training: Late shifts the selected layers deeper into the network. They are experimental variants, not successive releases or a ranking of model quality. **Three-domain** adds an instruction-following specialist to math and coding supervision.

Each release is a complete model with its tokenizer, ready to load with Transformers; teacher models are needed only for training. See [use a released model](docs/inference.md) for a working example, downloads, and fixed release revisions.

## Getting started

To use an existing model, start with the [inference guide](docs/inference.md). The steps below are for training OPTD.

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
