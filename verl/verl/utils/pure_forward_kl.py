# Copyright 2026 OPRD contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Pure full-vocabulary forward KL, with optional frozen intermediate JLens.

Final loss uses ALL response prediction positions, intermediate loss last2000.
There is no score-function/PPO/reward loss. Teacher hidden states, not vocabulary
logits, cross Ray; both heads are streamed with checkpointed readout recomputation.
"""
from pathlib import Path
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from verl.utils.jlens_distillation import JLensArtifacts, jlens_transported_residuals
from verl.utils.rep_distillation import build_rep_distillation_position_mask


def prediction_hidden(model, inputs, layer=20):
    """Slot0 is post-block20; slot1 is already-final-normalized. Shift by one."""
    with torch.autocast(device_type=inputs['input_ids'].device.type, dtype=torch.bfloat16):
        out = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    position_ids=inputs['position_ids'], use_cache=False,
                    output_hidden_states=True, logits_to_keep=1)
    r = inputs['response_mask'].shape[1]
    assert len(out.hidden_states) > layer + 2
    return torch.stack([out.hidden_states[layer + 1][:, -r-1:-1],
                        out.hidden_states[-1][:, -r-1:-1]], dim=1)


def load_teacher_readout(model_path, device):
    """Load only the frozen Qwen2 final norm/head from original teacher weights."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    root = Path(model_path)
    cfg = AutoConfig.from_pretrained(root, local_files_only=True)
    if cfg.model_type != 'qwen2':
        raise ValueError('This controlled experiment is restricted to the Qwen2 1.5B pair')
    index = root / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else {}
    def tensor(key):
        file = root / mapping.get(key, 'model.safetensors')
        with safe_open(file, framework='pt', device='cpu') as f:
            return f.get_tensor(key)
    norm = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight = nn.Parameter(tensor('model.norm.weight').to(device), requires_grad=False)
    head_key = 'model.embed_tokens.weight' if cfg.tie_word_embeddings else 'lm_head.weight'
    head = tensor(head_key).to(device=device, dtype=torch.float32)
    return norm, head


def streamed_kl(student_hidden, teacher_hidden, student_head, teacher_head, mask, chunk_size=32):
    """Token-mean KL(T||S); full vocabulary, T=1, frozen teacher, exact gradients."""
    valid = mask.bool()
    s = student_hidden[valid].float()
    t = teacher_hidden.detach()[valid].float()
    if not s.shape[0]:
        raise ValueError('Empty forward KL mask')
    # One differentiable, independently owned full head copy, safe across FSDP reshard.
    sw = student_head.to(dtype=torch.float32, copy=True)
    tw = teacher_head.detach()
    def block(hs, ht, ws, wt):
        sl = F.log_softmax(F.linear(hs, ws), dim=-1)
        with torch.no_grad():
            tl = F.log_softmax(F.linear(ht, wt), dim=-1)
            tp = tl.exp()
        return (tp * (tl - sl)).sum()
    total = s.new_zeros(())
    for start in range(0, s.shape[0], chunk_size):
        total = total + checkpoint(block, s[start:start+chunk_size], t[start:start+chunk_size],
                                   sw, tw, use_reentrant=False)
    return total / s.shape[0]


def pure_forward_kl_loss(student, teacher, mask, student_norm, student_head, teacher_norm,
                         teacher_head, student_lens=None, teacher_lens=None, last_k=2000, final_only=False):
    if (student_lens is None) != (teacher_lens is None):
        raise ValueError('Both lenses or neither lens must be supplied')
    if final_only and student_lens is not None:
        raise ValueError('Final-only KL cannot use intermediate lenses')
    intermediate = student.new_zeros(())
    mid_mask = torch.zeros_like(mask)
    if not final_only:
        mid_s, mid_t = student[:, :1].float(), teacher[:, :1].detach().float()
        if student_lens is not None:
            mid_s = jlens_transported_residuals(mid_s, artifacts=student_lens)
            mid_t = jlens_transported_residuals(mid_t, artifacts=teacher_lens)
        mid_s = student_norm(mid_s[:, 0]).float()
        with torch.no_grad():
            mid_t = teacher_norm(mid_t[:, 0]).float()
        mid_mask = build_rep_distillation_position_mask(mask, 'last_k', last_k=last_k)
        intermediate = streamed_kl(mid_s, mid_t, student_head, teacher_head, mid_mask)
    # HF hidden_states[-1] has ALREADY passed final RMSNorm: do not apply it twice.
    final = streamed_kl(student[:, 1], teacher[:, 1], student_head, teacher_head, mask)
    loss = final + intermediate
    return loss, {'pure_fkl/final': final.detach(), 'pure_fkl/intermediate': intermediate.detach(),
                  'pure_fkl/total': loss.detach(), 'pure_fkl/final_tokens': mask.sum().detach(),
                  'pure_fkl/intermediate_tokens': mid_mask.sum().detach(),
                  'pure_fkl/reward_opd_weight': loss.new_zeros(()),
                  'pure_fkl/full_vocab': loss.new_ones(())}
