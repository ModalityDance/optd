"""Verify same-token-ID R1 7B models and complete content-addressed weight blobs."""
import hashlib
import json
import sys
from pathlib import Path

reference = None
for raw in sys.argv[1:]:
    p = Path(raw)
    c = json.loads((p/'config.json').read_text())
    assert (c['model_type'], c['hidden_size'], c['num_hidden_layers'], c['vocab_size']) == ('qwen2', 3584, 28, 152064)
    t = json.loads((p/'tokenizer.json').read_text())
    sig = (t['model']['vocab'], [tuple(x.split(' ',1) if isinstance(x,str) else x) for x in t['model']['merges']], t['added_tokens'])
    if reference is None:
        reference = sig
    assert sig == reference, f'Token IDs differ: {p}'
    for name in set(json.loads((p/'model.safetensors.index.json').read_text())['weight_map'].values()):
        f = p/name
        blob = f.resolve().name
        assert len(blob) == 64, f'Expected HF LFS SHA256 blob: {f}'
        with f.open('rb') as h:
            assert hashlib.file_digest(h,'sha256').hexdigest() == blob, f'Corrupt weight: {f}'
    print('R1DISTILL7B_PREFLIGHT_OK',p,flush=True)
