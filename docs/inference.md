# Use a released OPTD model

The [README inference section](../README.md#inference) is the complete runbook: every checkpoint download, both generation modes, and every paper dataset. This note adds the release revisions, checksums, and template details behind those commands.

Each of the six models contains student weights and a tokenizer. Inference does not require a teacher or a training job.

## Generation mode by size

| Checkpoint | Hub repository | Generation mode |
| --- | --- | --- |
| 1.5B Mid | [OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Mid](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Mid) | Reasoning template |
| 1.5B Late | [OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Late](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-Late) | Reasoning template |
| 1.5B Three-domain | [OPTD-DeepSeek-R1-Distill-Qwen-1.5B-ThreeDomain](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-1.5B-ThreeDomain) | Reasoning template |
| 4B Mid | [OPTD-Qwen3-4B-Mid](https://huggingface.co/ModalityDance/OPTD-Qwen3-4B-Mid) | `enable_thinking=False` |
| 4B Late | [OPTD-Qwen3-4B-Late](https://huggingface.co/ModalityDance/OPTD-Qwen3-4B-Late) | `enable_thinking=False` |
| 7B Late | [OPTD-DeepSeek-R1-Distill-Qwen-7B-Late](https://huggingface.co/ModalityDance/OPTD-DeepSeek-R1-Distill-Qwen-7B-Late) | Reasoning template |

For either Qwen3-4B model, keep `enable_thinking=False`. For an R1-Distill model, omit that argument and use the tokenizer's native reasoning template. The model's tokenizer supplies the chat format. When the example first formats the chat as text, `add_special_tokens=False` avoids adding special tokens a second time.

The 1,024-token limit in the README example is for a short demonstration. Dataset runs use the token limits in the [evaluation guide](evaluation.md).

## Pinned downloads

[released_checkpoints.json](../configs/released_checkpoints.json) records the original release commit for each model. The README download block already passes these revisions. To load a revision without a local directory, pass it to both `from_pretrained` calls:

```python
revision = "c628ec8524a1564d62b83c163c0f2c479c7216f8"  # Qwen3-4B Late
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
)
```

| Model | Revision |
| --- | --- |
| 1.5B Mid | `800ec92fe578870e74df401a400a2b51a801ac99` |
| 1.5B Late | `57241e09a7ebd5a36c80897c69c17fd9a0ae7901` |
| 1.5B Three-domain | `12485f39c4628521ff6f784978a176003197a7df` |
| 4B Mid | `45e4f740ee974959c99ef6e9ab97a1ec2b386fb5` |
| 4B Late | `c628ec8524a1564d62b83c163c0f2c479c7216f8` |
| 7B Late | `e9b49f3c9b33fd3667808cd4eafa6512176051a9` |

Each model repository includes `training_metadata.json` and `checksums.json`, which records SHA256 hashes for the checkpoint and tokenizer files. Releases contain inference weights rather than optimizer state; they do not restore the complete state of an interrupted training run.

## Dataset entry points

The README lists the full command for each dataset and the flag that changes with model size. Those flags are:

| Dataset | Qwen3-4B Mid / Late | R1-Distill 1.5B / 7B |
| --- | --- | --- |
| AIME24, AIME25, AIMO | `--no-enable-thinking` on `eval_checkpoint_vllm.py` | thinking left on (script default) |
| LiveCodeBench v5 | `--non-thinking` on `run_lcb_eval.py` | omit `--non-thinking` |
| CodeContests-165 | `--non-thinking` on `run_deepmind_codecontests165_eval.py` | omit `--non-thinking` |
| Codeforces | `eval_codeforces_codeelo_vllm.py` always sets `enable_thinking=False` | same script, same template |
| IFEval | original checker, non-thinking prompt | original checker, reasoning template |

Math parquet files are `datasets/test_data/{AIME24,AIME25,AIMO}/test.parquet`. LiveCodeBench, CodeContests, and Codeforces files are external inputs; the commands in the README take their paths explicitly. IFEval grading stays on the original benchmark checker.
