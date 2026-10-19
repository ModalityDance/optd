"""Merge two immutable n4 batches, retaining responses and verified verdicts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def batch(generation, graded):
    rows, verdicts = read(generation), read(graded)
    assert len(rows) == 279 and len(verdicts) == 1116
    by_id = {r['example_id']: r for r in rows}
    assert len(by_id) == 279
    scores = {}
    for v in verdicts:
        eid, index = v['example_id'], int(v['key'].rsplit(':', 1)[1])
        assert eid in by_id and 0 <= index < 4 and (eid, index) not in scores
        assert v['score'] in (0, 1)
        response = by_id[eid]['responses'][index]
        assert hashlib.sha256(response.encode()).hexdigest() == v['sha256']
        scores[eid, index] = v
    result = {}
    for eid, row in by_id.items():
        assert len(row['responses']) == 4
        result[eid] = [dict(response=response, verdict=scores[eid, i])
                       for i, response in enumerate(row['responses'])]
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('first_generation', 'first_graded', 'second_generation', 'second_graded', 'output_dir'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    one = batch(args.first_generation, args.first_graded)
    two = batch(args.second_generation, args.second_graded)
    assert one.keys() == two.keys()
    merged = []
    for eid in sorted(one):
        candidates = [dict(batch=b, candidate_index=i, **c)
                      for b, items in enumerate((one[eid], two[eid])) for i, c in enumerate(items)]
        merged.append(dict(example_id=eid, candidates=candidates))
    def summary(groups):
        scores = [[c['verdict']['score'] for c in group] for group in groups]
        n = len(scores[0])
        passed, solved = sum(map(sum, scores)), sum(map(any, scores))
        return dict(problems=279, candidates=279*n, passed_candidates=passed,
                    solved_problems=solved, **{f'mean_at_{n}_percent':100*passed/(279*n),
                                              f'pass_at_{n}_percent':100*solved/279})
    result = summary([r['candidates'] for r in merged])
    result['first_n4'] = summary(list(one.values()))
    result['second_n4'] = summary(list(two.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'lcb_n8_merged.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in merged))
    sources = {k:dict(path=str(v.resolve()), sha256=hashlib.sha256(v.read_bytes()).hexdigest())
               for k,v in vars(args).items() if k != 'output_dir'}
    result['sources'] = sources
    (args.output_dir/'lcb_n8.summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
