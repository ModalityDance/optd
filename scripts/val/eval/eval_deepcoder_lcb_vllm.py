#!/usr/bin/env python3
"""Generate the released DeepCoder single-turn LCB-v5 qualification pass."""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams

from deepcoder_lcb_common import discover_lcb_glob, format_rllm_deepcoder_chat_prompt, load_lcb_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-glob")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # Official rLLM/OpenAI request does not set a per-request sampling seed.
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="number of candidates per problem from this single vLLM instance",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--non-thinking", action="store_true")
    args = parser.parse_args()
    if args.data_glob is None:
        args.data_glob = discover_lcb_glob()
        print(f"Discovered validated LCB dataset: {args.data_glob}", flush=True)
    print(f"non_thinking={args.non_thinking}", flush=True)

    rows = load_lcb_rows(args.data_glob)
    if args.start < 0 or args.start >= len(rows) or (args.end is not None and args.end <= args.start):
        parser.error("invalid --start/--end range")
    rows = rows[args.start : args.end]
    existing = {}
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                key = item.get("example_id")
                if key in existing:
                    raise ValueError(f"duplicate existing example_id: {key}")
                if not isinstance(item.get("responses"), list) or len(item["responses"]) != args.n:
                    raise ValueError(f"existing record {key} does not contain n={args.n} responses")
                existing[key] = item
        expected_ids = {row["example_id"] for row in rows}
        unknown = set(existing) - expected_ids
        if unknown:
            raise ValueError(f"existing output contains IDs outside requested range: {sorted(unknown)[:3]}")
    pending_rows = [row for row in rows if row["example_id"] not in existing]
    print(f"Loaded {len(rows)} LCB-v5 test problems (rows {args.start}:{args.end}); "
          f"reuse={len(existing)}, generate={len(pending_rows)}", flush=True)

    if not pending_rows:
        print(f"All requested problems already exist in {args.output}; skipping generation", flush=True)
        return

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.n < 1:
        parser.error("--n must be >= 1")
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=args.n,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [format_rllm_deepcoder_chat_prompt(tokenizer, row["question"], non_thinking=args.non_thinking) for row in pending_rows]
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        generated = {}
        for row, output in zip(pending_rows, outputs, strict=True):
            candidates = output.outputs
            generated[row["example_id"]] = {
                        "example_id": row["example_id"],
                        "source_path": row["source_path"],
                        "source_index": row["source_index"],
                        "responses": [candidate.text for candidate in candidates],
                        "finish_reasons": [candidate.finish_reason for candidate in candidates],
                        "response_tokens": [len(candidate.token_ids) for candidate in candidates],
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "n": args.n,
                        "seed": args.seed,
                        "agent_harness": "qwen3.native.non_thinking" if args.non_thinking else "rllm.DeepseekQwenChatTemplateParser",
            }
        for row in rows:
            item = existing[row["example_id"]] if row["example_id"] in existing else generated[row["example_id"]]
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} problems x {args.n} candidates to {args.output} "
          f"({len(existing)} reused, {len(generated)} generated)", flush=True)


if __name__ == "__main__":
    main()
