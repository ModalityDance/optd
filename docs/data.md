# Models and data

Weights and datasets are external prerequisites. The table below records the model identifiers used by the manuscript; download and verify the intended revisions before running comparisons. Public model IDs alone do not guarantee the exact checkpoint/tokenizer preparation used by the experiments.

| Size | Student | Math specialist | Coding specialist |
| --- | --- | --- | --- |
| 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `hbx/JustRL-DeepSeek-1.5B` | `agentica-org/DeepCoder-1.5B-Preview` |
| 4B | `Qwen/Qwen3-4B` | `pb09204048/Qwen3-4B-Math-RL-Specialist` | `pb09204048/Qwen3-4B-Code-RL-Specialist` |
| 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `Skywork/Skywork-OR1-Math-7B` | `nvidia/AceReason-Nemotron-7B` |

The three-domain 1.5B extension adds `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` (ProRL v2) as the instruction-following teacher. Export its local path as `TERTIARY_REWARD_MODEL_PATH`.

Teacher and student candidate IDs must refer to the same tokens. Preserve the preflight checks in `scripts/preflight_qwen4b_teachers.py` and `scripts/preflight_r1distill7b_teachers.py`. The 7B preflight expects Hugging Face cache symlinks resolving to SHA256-named weight blobs; preserve that cache layout when supplying local paths. The 1.5B driver additionally checks the expected DeepCoder weight shards and tokenizer vocabulary/merges.

## Training datasets

- Math: `TURLEing/DAPO-Math-17k`, converted to `datasets/dapo-math-17k.parquet`.
- Coding: `agentica-org/DeepCoder-Preview-Dataset`, pinned to revision `177913a7bd43791646ef6a43645caa3c871ab3db`. Include all train shards from taco, primeintellect, and lcbv5.
- Instruction following: `TURLEing/Rubrics-To-Tokens`, used by the three-domain extension.

The expected training row contains `data_source`, a user `prompt` message list, `ability`, `reward_model` (`style`, `ground_truth`), and `extra_info`. Coding test cases are normalized for SandboxFusion; the data builder does not execute candidate code. Math and coding source labels are `math_dapo` and `codecontests`; the third domain is `instruction_following`.

After obtaining the inputs, build the full Math+Code pool:

```bash
python scripts/build_multiteacher_math_code_data.py \
  --math datasets/dapo-math-17k.parquet \
  --code-root /path/to/DeepCoder-Preview-Dataset \
  --pool-mode full_pool \
  --output datasets/math_code_full_pool.parquet
```

The builder creates a sibling `.manifest.json` with counts/checksum information. Retain it with the parquet. The 4B/7B profiles require the full pool with balanced-cycle domain sampling; use the built path for `TRAIN_DATASET`. For the historical 1.5B fixed-2,400 schedule, use `--pool-mode fixed_2400 --steps 150` and a separate output path.

The three-domain converter takes an existing fixed Math+Code schedule and the original Rubrics-To-Tokens JSONL:

```bash
python scripts/build_three_teacher_hir_data.py \
  --math-code /path/to/fixed_math_code.parquet \
  --hir /path/to/rubrics_to_tokens.jsonl \
  --output datasets/math_code_if.parquet \
  --manifest datasets/math_code_if.parquet.manifest.json
```

This converter enforces the original input checksum and 16,968-row IF source contract. If those exact inputs are unavailable, the source bundle alone cannot reconstruct the historical three-domain sample. Do not remove the assertions and describe a different sample as the original experiment.

## Validation files

The inherited training driver expects these separately prepared benchmark files:

```text
datasets/test_data/AMC23/test.parquet
datasets/test_data/AIME24/test.parquet
datasets/test_data/AIME25/test.parquet
datasets/test_data/AIMO/test.parquet
datasets/test_data/CodeForces-DeepCoder-Harness/test.parquet
```

Use the same prompt/reward row schema, source labels, and test normalization as the training builder. `scripts/preflight_multiteacher_validation.py` checks reward dispatch and the SandboxFusion contract. Disabling periodic validation does not remove the inherited preflight artifact requirements.

No model checkpoints, datasets, private download links, or cluster access bundles are included in this repository. Dataset conversion and model compatibility checks are included; downloading and preparing every upstream benchmark is still required.
