# Evaluation

Use a merged Hugging Face actor checkpoint with the student tokenizer. The vendored merger is available through `python -m verl.model_merger --help`. Do not compare a different training step or prompt template to a paper result. Generation uses vLLM; coding grading requires a separately running SandboxFusion service at its `/run_code` endpoint.

| Task | Paper reporting | Generation |
| --- | --- | --- |
| AIME24, AIME25, AIMO | Avg@16; corresponding pass scores | n=16, T=0.7, top-p=0.95, 31,744 tokens |
| LiveCodeBench v5 | Avg@8 and Pass@8 over 279 problems | Merge two independent n=4 batches |
| CodeContests | Avg@16 over the 165-problem subset | n=16 |
| Codeforces | Local CodeElo estimate plus Avg@8/Pass@8 | 408 canonical problems, n=8 |
| IFEval | Mean strict/loose accuracy across 16 samples | Original IF evaluation harness required |

## Math

```bash
PYTHONPATH="$PWD/verl" python scripts/val/eval/eval_checkpoint_vllm.py \
  --model-path /path/to/merged_actor \
  --data-dir datasets/test_data \
  --output-dir outputs/math \
  --tasks AIME24 AIME25 AIMO --n 16
```

For Qwen3-4B, add `--no-enable-thinking`. The script saves generations. Run rule-based grading separately and retain per-question outcomes:

```bash
PYTHONPATH="$PWD/verl" python scripts/val/eval/grade.py --eval-dir outputs/math
```

The optional model-verifier branch is not enabled by this command. Match the paper's grading protocol when reporting a comparison.

## LiveCodeBench

```bash
PYTHONPATH="$PWD/verl" python scripts/val/eval/run_lcb_eval.py \
  --model-path /path/to/merged_actor \
  --data-glob '/path/to/lcbv5/test-*.parquet' \
  --output-prefix outputs/lcb/batch_a \
  --sandbox-url http://127.0.0.1:8080/run_code
```

Run a second independent n=4 batch into `batch_b`; use `--non-thinking` for Qwen3-4B. Merge generations and verified verdicts together:

```bash
python scripts/val/eval/merge_lcb_n8_complete.py \
  --first-generation outputs/lcb/batch_a.jsonl \
  --first-graded outputs/lcb/batch_a.graded.jsonl \
  --second-generation outputs/lcb/batch_b.jsonl \
  --second-graded outputs/lcb/batch_b.graded.jsonl \
  --output-dir outputs/lcb/n8
```

The merge checks 279 unique problems, 1,116 candidates per n=4 batch, response hashes, and verdict coverage. Compute Avg@8/Pass@8 from the merged eight verdicts, rather than averaging Pass@4. Both batches must be independently generated under the same protocol.

## CodeContests and CodeElo

```bash
PYTHONPATH="$PWD/verl" python scripts/val/eval/run_deepmind_codecontests165_eval.py \
  --model-path /path/to/merged_actor \
  --data /path/to/codecontests165.parquet \
  --output-prefix outputs/codecontests/run --n 16 \
  --sandbox-url http://127.0.0.1:8080/run_code
```

Add `--non-thinking` for Qwen3-4B. Verify 165 problems and 2,640 generated/graded candidates. The Codeforces generation helper is `eval_codeforces_codeelo_vllm.py`; the original canonical problem metadata, contest mapping, historical ratings, grading harness, and Elo estimator are external prerequisites. An Elo estimate requires the complete matched pipeline and cannot be obtained from generation output alone.

## Instruction following and scope

The source bundle includes the training data route for the IF specialist. It does not include a standalone end-to-end IFEval release harness or the original evaluation artifacts. Use the original benchmark checker and record strict/loose prompt/instruction metrics with n=16 before comparing to the paper.

## Execution and validation

- Select a checkpoint explicitly and record the exact dataset revision, candidate count, template mode, generation seed/settings, and grading harness.
- For Qwen3-4B, use the tokenizer's native `enable_thinking=False` and inspect representative prompts/outputs for unintended `<think>` content.
- Generation and grading are separate queues: once GPU generation finishes and GPUs are free, the next generation can start while CPU grading continues. The convenience wrappers are sequential; use the underlying generation/grading scripts to overlap stages.
- Preserve partial generation artifacts and failed grading outcomes. Report scores only after unique IDs and expected coverage are verified.
- Stop services started for evaluation when no active job needs them.
