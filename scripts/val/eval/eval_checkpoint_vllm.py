#!/usr/bin/env python3
"""Evaluate one local checkpoint with the repository's Avg@N generation protocol."""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."


def load_samples(path: Path):
    df = pd.read_parquet(path)
    samples = []
    for index in range(len(df)):
        samples.append(
            {
                "example_id": index,
                "prompt": df.at[index, "prompt"][0]["content"].strip(),
                "answer": df.at[index, "reward_model"]["ground_truth"].strip(),
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/test_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["AIME24", "AIME25", "AIMO"])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"enable_thinking={args.enable_thinking}; existing task files are reused unless --overwrite is set", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded:
            stop_token_ids.append(encoded[0])
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids or None,
    )

    pending = []
    for task in args.tasks:
        out_path = args.output_dir / (
            f"{task.lower()}_t{args.temperature}_p{args.top_p}_n{args.n}-MNT{args.max_tokens}.jsonl"
        )
        if out_path.exists() and not args.overwrite:
            print(f"{task}: skip-existing {out_path} (reusing saved responses)", flush=True)
            continue
        samples = load_samples(args.data_dir / task / "test.parquet")
        requests, metadata = [], []
        for rollout_id in range(args.n):
            for sample in samples:
                prompt = PROMPT_TEMPLATE.format(problem=sample["prompt"])
                requests.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                )
                metadata.append((sample, rollout_id, prompt))
        print(f"{task}: queued {len(requests)} responses ({len(samples)} problems x {args.n})", flush=True)
        pending.append((task, out_path, requests, metadata))

    if not pending:
        return

    all_requests = []
    offsets = []
    for task, out_path, requests, metadata in pending:
        offsets.append((task, out_path, metadata, len(all_requests), len(all_requests) + len(requests)))
        all_requests.extend(requests)
    print(f"Generating {len(all_requests)} responses across {len(pending)} tasks in one vLLM request batch", flush=True)
    all_outputs = llm.generate(all_requests, sampling, use_tqdm=True)
    for task, out_path, metadata, start, end in offsets:
        with out_path.open("w", encoding="utf-8") as f:
            for output, (sample, rollout_id, prompt) in zip(all_outputs[start:end], metadata):
                f.write(
                    json.dumps(
                        {
                            "example_id": sample["example_id"],
                            "prompt": prompt,
                            "answer": sample["answer"],
                            "seed": rollout_id,
                            "response": output.outputs[0].text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
