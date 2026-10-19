"""Frozen endpoint-union support for depth-transition distillation."""
import torch
from torch import nn


def path_readout(hidden, layers, ids, norm, head, chunk):
    from verl.utils.jlens_distillation import layerwise_log_probs_on_policy_topk
    pieces = []
    for layer in [*layers, hidden.size(1) - 1]:
        # HF's final hidden state has already passed through final RMSNorm.
        pieces.append(layerwise_log_probs_on_policy_topk(
            hidden[:, layer:layer + 1], policy_topk_ids=ids,
            final_norm=nn.Identity() if layer == hidden.size(1) - 1 else norm,
            lm_head=head, token_chunk_size=chunk, temperature=1.0))
    return torch.cat(pieces, dim=1)


def snapshot(hidden, layers, norm, head, chunk):
    from verl.utils.jlens_distillation import layerwise_topk_log_probs
    top = []
    for layer in [*layers, hidden.size(1) - 1]:
        ids, _ = layerwise_topk_log_probs(
            hidden[:, layer:layer + 1], top_k=32,
            final_norm=nn.Identity() if layer == hidden.size(1) - 1 else norm,
            lm_head=head, token_chunk_size=chunk, temperature=1.0)
        top.append(ids[:, 0])
    # Fixed-width transport; duplicates are masked before ranking each pair.
    ids = torch.cat(top, dim=-1)
    return ids, path_readout(hidden, layers, ids, norm, head, chunk)


def select_indices(ids, student, teacher, mode):
    if mode not in {"max", "min"}:
        raise ValueError(mode)
    if student.shape != teacher.shape or student.shape[1] != 4 or ids.size(-1) != 128:
        raise ValueError("expected four endpoint readouts on concatenated Top32 ids")
    error = ((student[:, 1:] - student[:, :-1]) -
             (teacher[:, 1:] - teacher[:, :-1])).detach().abs()
    chosen = []
    for pair in range(3):
        start = pair * 32
        candidates = ids[..., start:start + 64]
        same = candidates.unsqueeze(-1) == candidates.unsqueeze(-2)
        earlier = torch.ones(64, 64, device=ids.device, dtype=torch.bool).tril(-1)
        duplicate = (same & earlier).any(-1)
        scores = error[:, pair, :, start:start + 64]
        scores = scores.masked_fill(duplicate, -torch.inf if mode == "max" else torch.inf)
        chosen.append(scores.topk(16, dim=-1, largest=mode == "max").indices + start)
    return torch.stack(chosen, dim=1)


def losses(current, teacher, indices, mask):
    src = current[:, :-1].gather(-1, indices)
    dst = current[:, 1:].gather(-1, indices)
    target = (teacher[:, 1:] - teacher[:, :-1]).gather(-1, indices).detach()
    delta = dst - src
    coeff = (dst.detach().softmax(-1) * (delta.detach() - target)).detach()
    token_loss = (coeff * delta).sum(-1)
    return (token_loss * mask[:, None]).sum((0, 2)) / mask.sum().clamp_min(1)
