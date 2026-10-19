#!/usr/bin/env python3
"""Generate Python solutions for the pinned DeepMind CodeContests test split."""

import argparse
import json
from pathlib import Path

from deepmind_codecontests165_common import (
    format_rllm_deepcoder_chat_prompt,
    load_deepmind_codecontests165_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=15360)
    parser.add_argument("--max-model-len", type=int, default=17408)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--non-thinking", action="store_true")
    args = parser.parse_args()
    print(f"non_thinking={args.non_thinking}", flush=True)
    if args.n < 1:
        parser.error("--n must be positive")

    from vllm import LLM, SamplingParams

    rows = load_deepmind_codecontests165_rows(args.data)
    existing = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                key = item["example_id"]
                if key in existing or len(item.get("responses", [])) != args.n:
                    raise ValueError(f"invalid existing generation record: {key}")
                existing[key] = item
    expected = {row["example_id"] for row in rows}
    if set(existing) - expected:
        raise ValueError("generation file contains IDs outside the pinned test split")
    pending = [row for row in rows if row["example_id"] not in existing]
    if pending:
        llm = LLM(model=args.model_path, trust_remote_code=True,
                  tensor_parallel_size=args.tensor_parallel_size,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization)
        tokenizer = llm.get_tokenizer()
        prompts = [format_rllm_deepcoder_chat_prompt(tokenizer, row["question"], non_thinking=args.non_thinking) for row in pending]
        sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                  max_tokens=args.max_tokens, n=args.n, seed=args.seed)
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        for row, output in zip(pending, outputs, strict=True):
            existing[row["example_id"]] = {
                "example_id": row["example_id"], "name": row["name"],
                "source_index": row["source_index"],
                "responses": [candidate.text for candidate in output.outputs],
                "finish_reasons": [candidate.finish_reason for candidate in output.outputs],
                "response_tokens": [len(candidate.token_ids) for candidate in output.outputs],
                "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens, "n": args.n, "seed": args.seed,
                "agent_harness": "qwen3.native.non_thinking" if args.non_thinking else "rllm.DeepseekQwenChatTemplateParser",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(existing[row["example_id"]], ensure_ascii=False) + "\n")
    print(f"DeepMind CodeContests165 generations complete: {len(rows)} problems x {args.n}", flush=True)


if __name__ == "__main__":
    main()
