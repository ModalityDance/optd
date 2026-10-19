"""Fail closed on Qwen3 specialist weights and shared tokenizer IDs."""
import hashlib
import json
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
assert len(paths) == 3
reference = None
for index, path in enumerate(paths):
    tokenizer = json.loads((path / 'tokenizer.json').read_text())
    cfg = json.loads((path / 'config.json').read_text())
    assert cfg['model_type'] == 'qwen3'
    signature = (tokenizer['model']['vocab'],
                 [tuple(x.split(' ', 1) if isinstance(x, str) else x)
                  for x in tokenizer['model']['merges']], tokenizer['added_tokens'])
    if reference is None:
        reference = signature
    else:
        assert signature == reference, f'Tokenizer mismatch: {path}'
    if index:
        provenance = json.loads((path / 'model_provenance.json').read_text())
        hashes = provenance['hf_file_sha256']
        weights = set(json.loads((path / 'model.safetensors.index.json').read_text())['weight_map'].values())
        assert weights <= hashes.keys()
        for name in weights:
            with (path / name).open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == hashes[name], name
    print('QWEN4B_PREFLIGHT_OK', path, flush=True)
