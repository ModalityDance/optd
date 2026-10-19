"""Generate eight ordered Codeforces candidates with the existing Qwen3 code prompt."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--tokenizer", type=Path, help="Explicit qualification tokenizer; defaults to model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260928)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo / "scripts/val/eval"))
    from deepcoder_lcb_common import load_lcb_rows
    from vllm import LLM, SamplingParams

    rows = load_lcb_rows(a.data_glob)
    assert len(rows) == 408 and len({r["example_id"] for r in rows}) == 408
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": str(a.model), "data_glob": a.data_glob,
        "data_sha256": None,
        "seed": a.seed, "thinking": False, "dtype": "bfloat16",
        "tensor_parallel_size": a.tensor_parallel_size, "max_model_len": 40960,
        "n": 8, "temperature": 0.6, "top_p": 0.95, "top_k": -1,
        "max_tokens": 32768,
    }
    if a.tokenizer is not None:
        protocol["tokenizer"] = str(a.tokenizer)
    # Hash the exact shard independently of its absolute path.
    import glob
    shards = sorted(glob.glob(a.data_glob))
    assert len(shards) == 1, shards
    protocol["data_sha256"] = hashlib.sha256(Path(shards[0]).read_bytes()).hexdigest()
    manifest = a.output / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == protocol, "protocol changed"
    else:
        manifest.write_text(json.dumps(protocol, indent=2) + "\n")

    llm = LLM(model=str(a.model), tokenizer=str(a.tokenizer or a.model), dtype="bfloat16",
              tensor_parallel_size=a.tensor_parallel_size, max_model_len=40960,
              gpu_memory_utilization=0.85, seed=a.seed,
              generation_config="vllm")
    tok = llm.get_tokenizer()
    chunks = a.output / "batches"
    chunks.mkdir(exist_ok=True)
    batches = []
    pending_prompts, pending_sampling = [], []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        path = chunks / f"{start:04d}.json"
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for r in batch]
        prompt_ids = [tok.encode(prompt, add_special_tokens=False) for prompt in prompts]
        assert all(len(ids) + 32768 <= 40960 for ids in prompt_ids)
        digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
        saved = None
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["prompt_sha256"] == digest
        else:
            pending_prompts.extend(prompts)
            pending_sampling.extend([SamplingParams(
                n=8, temperature=0.6, top_p=0.95, top_k=-1,
                max_tokens=32768, seed=a.seed + start,
                stop_token_ids=[tok.eos_token_id]) for _ in batch])
        batches.append((start, batch, path, digest, saved))
    print(f"SUBMIT_PENDING {len(pending_prompts)} problems", flush=True)
    outputs = iter(llm.generate(pending_prompts, pending_sampling, use_tqdm=True)
                   if pending_prompts else [])
    all_records = []
    for start, batch, path, digest, saved in batches:
        if saved is not None:
            records = saved["records"]
        else:
            records = []
            for row in batch:
                output = next(outputs)
                assert len(output.outputs) == 8
                records.append({
                    "example_id": row["example_id"],
                    "source_index": row["source_index"],
                    "responses": [c.text for c in output.outputs],
                    "response_tokens": [len(c.token_ids) for c in output.outputs],
                    "finish_reasons": [c.finish_reason for c in output.outputs],
                })
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"prompt_sha256": digest, "records": records}, ensure_ascii=False))
            temporary.replace(path)
        all_records.extend(records)
        print(f"BATCH_OK {min(start + 8, len(rows))}/{len(rows)}", flush=True)
    assert len(all_records) == 408
    final = a.output / "codeforces_n8.jsonl"
    temporary = final.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records))
    temporary.replace(final)
    print("GENERATION_COMPLETE 408x8", flush=True)


if __name__ == "__main__":
    main()
