#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--first',type=Path,required=True); p.add_argument('--second',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
def load(path):
 rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
 if len(rows)!=1116: raise SystemExit(f'{path}: expected 1116, got {len(rows)}')
 out={}
 for r in rows:
  if r['key'] in out: raise SystemExit(f'duplicate key {r["key"]}')
  out[r['key']]=r
 return out
one,two=load(a.first),load(a.second); ids=sorted({r['example_id'] for r in one.values()})
if len(ids)!=279 or {r['example_id'] for r in two.values()} != set(ids): raise SystemExit('coverage mismatch')
merged=[]; passed=solved=0
for eid in ids:
 rs=sorted([r for r in one.values() if r['example_id']==eid]+[r for r in two.values() if r['example_id']==eid], key=lambda r:(r['key'].split(':')[0],int(r['key'].rsplit(':',1)[1])))
 scores=[int(r['score']) for r in rs]; passed+=sum(scores); solved+=int(any(scores)); merged.append({'example_id':eid,'scores':scores,'passed_candidates':sum(scores),'solved':bool(any(scores)),'keys':[r['key'] for r in rs]})
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.with_suffix('.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in merged)); summary={'problems':279,'candidates':2232,'passed_candidates':passed,'solved_problems':solved,'mean_at_8_percent':100*passed/2232,'pass_at_8_percent':100*solved/279}; a.output.write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary))
