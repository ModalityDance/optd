"""Binary LCB grading with bounded concurrency and resumable candidate records.

Uses the existing per-case runner/comparator. A failed test suffices to fail
an all-tests-pass candidate; infrastructure errors abort rather than score zero.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

from deepcoder_lcb_common import discover_lcb_glob, load_lcb_rows
from verl.utils.reward_score.sandbox_fusion import utils as sandbox_utils

# Large LCB stdin payloads can exceed the client's default HTTP deadline while
# the sandbox still enforces the original 6-second execution limit.
sandbox_utils.API_TIMEOUT = 180


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generations', nargs='+', required=True)
    p.add_argument('--data-glob')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--sandbox-url', default='http://127.0.0.1:8080/run_code')
    a = p.parse_args()
    if a.data_glob is None:
        a.data_glob = discover_lcb_glob()
        print(f"Discovered validated LCB dataset: {a.data_glob}", flush=True)
    tests = {r['example_id']: r['tests'] for r in load_lcb_rows(a.data_glob)}
    jobs = []
    for file_index, filename in enumerate(a.generations):
        rows = [json.loads(line) for line in Path(filename).read_text().splitlines()]
        assert len(rows) == len(tests) and {r['example_id'] for r in rows} == set(tests)
        for row in rows:
            responses = row.get('responses', [row.get('response')])
            for index, response in enumerate(responses):
                assert isinstance(response, str)
                key = f'{file_index}:{row["example_id"]}:{index}'
                jobs.append((key, row['example_id'], response))
    counts = {eid: sum(j[1] == eid for j in jobs) for eid in tests}
    assert set(counts.values()) == {4}, counts
    old = {}
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            item = json.loads(line)
            assert item['key'] not in old
            old[item['key']] = item
    def grade(job):
        key, eid, response = job
        digest = hashlib.sha256(response.encode()).hexdigest()
        if key in old:
            assert old[key]['sha256'] == digest
            return old[key]
        blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        score, checked = 0, 0
        if blocks:
            t = tests[eid]
            assert len(t['inputs']) == len(t['outputs']) > 0
            score = 1
            for i, (inp, out) in enumerate(zip(t['inputs'], t['outputs'])):
                code = blocks[-1].strip() + '\n\n' + t.get('assert_case', [''] * len(t['inputs']))[i]
                result, meta = sandbox_utils._process_single_case(i, inp, out, a.sandbox_url, code, 6, 1024, 'python', None, t.get('fn_name'), True)
                if result == -1:
                    raise RuntimeError(f'infrastructure error: {key}, case {i}: {meta.get("status")}')
                checked += 1
                if result is not True:
                    score = 0
                    break
        return dict(key=key, example_id=eid, sha256=digest, score=score, tests_checked=checked)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    records = dict(old)
    with a.output.open('a') as handle, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(grade, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            if item['key'] not in records:
                handle.write(json.dumps(item) + '\n')
                handle.flush()
                records[item['key']] = item
            if len(records) % 100 == 0:
                print(f'{len(records)}/{len(jobs)} candidates graded', flush=True)
    assert len(records) == len(jobs)
    grouped = {eid: [records[j[0]]['score'] for j in jobs if j[1] == eid] for eid in tests}
    passed_candidates = sum(sum(v) for v in grouped.values())
    solved = sum(any(v) for v in grouped.values())
    summary = dict(problems=len(tests), candidates=len(jobs), passed_candidates=passed_candidates, solved_problems=solved, mean_at_4_percent=100 * passed_candidates/len(jobs), pass_at_4_percent=100 * solved/len(tests), protocol='existing per-case LCB comparator; binary fail-fast; 6s; 1024MiB')
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
