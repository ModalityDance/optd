#!/usr/bin/env python3
"""Build fixed 800/800/800 Math+Code+HIR mixture for the matched 150-step run."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20260923
SOURCE_SHA = "38a3626237e5758ccffb2f89355ce25c73ad23f778d0046a95ac580282358a0d"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    for name in ("math-code", "hir", "output", "manifest"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    assert sha(a.math_code) == SOURCE_SHA
    math_code = pq.read_table(a.math_code).to_pylist()
    hir = [json.loads(line) for line in a.hir.open()]
    assert len(hir) == 16968
    rng = random.Random(SEED)
    selected = {source: rng.sample([r for r in math_code if r["data_source"] == source], 800)
                for source in ("math_dapo", "codecontests")}
    indices = sorted(rng.sample(range(len(hir)), 800))
    selected["instruction_following"] = []
    for index in indices:
        row = hir[index]
        assert row["messages"] == [{"role": "user", "content": row["prompt"]}]
        selected["instruction_following"].append({
            "data_source": "instruction_following",
            "prompt": row["messages"],
            "ability": "IF",
            "reward_model": {"ground_truth": "", "style": "ifeval"},
            "extra_info": {"index": str(row["id"]), "source": "TURLEing/Rubrics-To-Tokens",
                           "source_index": str(index), "source_type": row["source"],
                           "question": row["question"],
                           "rubrics_json": json.dumps(row["rubrics"], ensure_ascii=False),
                           "ground_truth_json": json.dumps(row["ground_truth"], ensure_ascii=False)},
        })
    sources = ("math_dapo", "codecontests", "instruction_following")
    cursors = Counter()
    output = []
    for batch_index in range(150):
        batch = []
        for source_index, source in enumerate(sources):
            count = 6 if batch_index % 3 == source_index else 5
            batch.extend(selected[source][cursors[source]:cursors[source] + count])
            cursors[source] += count
        random.Random(SEED + batch_index).shuffle(batch)
        output.extend(batch)
    assert all(cursors[source] == 800 for source in sources)
    columns = ("data_source", "prompt", "ability", "reward_model", "extra_info")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{k: row[k] for k in columns} for row in output]), a.output, compression="zstd")
    manifest = {"rows": 2400, "steps": 150, "data_source_counts": dict(cursors), "seed": SEED,
                "hir_source_sha256": sha(a.hir), "hir_source_rows": len(hir),
                "hir_selected_source_indices": indices, "output_sha256": sha(a.output),
                "sampling": "uniform without replacement; mixed rotating 6/5/5 batches",
                "reward_semantics": "teacher_opd_only_no_rule_reward",
                "harness": "agentica-project/rllm DeepCoder single-turn prompt contract",
                "deepcoder_dataset_revision": "177913a7bd43791646ef6a43645caa3c871ab3db"}
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "data_source_counts", "output_sha256")}))

if __name__ == "__main__":
    main()
