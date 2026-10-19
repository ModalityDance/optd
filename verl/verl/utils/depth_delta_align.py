"""Detached token-wise weighting of existing depth transitions."""
import torch
import torch.nn.functional as F


def transition_alignment_weights(teacher_delta, student_delta):
    """Return [B,L,T] weights summing to one over L; all-zero uses uniform.

    Deltas are destination minus source log probabilities on the shared
    final-policy support. Routing never participates in autograd.
    """
    if teacher_delta.shape != student_delta.shape or teacher_delta.ndim != 4:
        raise ValueError("transition deltas must have matching [B,L,T,K] shapes")
    with torch.no_grad():
        score = F.cosine_similarity(
            teacher_delta.detach().float(), student_delta.detach().float(),
            dim=-1, eps=1e-6,
        ).clamp_min(0)
        total = score.sum(dim=1, keepdim=True)
        return torch.where(total > 0, score / total.clamp_min(1e-30),
                           torch.full_like(score, 1 / score.shape[1]))
