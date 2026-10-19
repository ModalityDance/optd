"""Token-level ExpertAlign routing for multi-teacher OPD."""

from __future__ import annotations

import torch


def expertalign_route_log_probs(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_base_topk_log_probs: torch.Tensor,
    *,
    weighting: str = "cosine",
    alignment_epsilon: float = 1e-6,
    norm_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route aligned teacher log-probabilities on the student's top-k support.

    ``teacher_topk_log_probs`` is teacher-major ``[N, B, T, K]``; the
    student and frozen shared-base tensors are ``[B, T, K]``.  A teacher is
    retained only when its base-relative specialization has positive raw
    inner product with its teacher-to-student correction.  Tokens for which
    no teacher survives use the student distribution, which yields exactly
    zero OPD advantage downstream.
    """
    if teacher_topk_log_probs.ndim != 4 or teacher_topk_log_probs.shape[0] < 2:
        raise ValueError("ExpertAlign requires at least two [N, B, T, K] teacher tensors")
    expected = teacher_topk_log_probs.shape[1:]
    if student_topk_log_probs.shape != expected:
        raise ValueError(
            f"student top-k shape {tuple(student_topk_log_probs.shape)} != {tuple(expected)}"
        )
    if teacher_base_topk_log_probs.shape != expected:
        raise ValueError(
            f"teacher-base top-k shape {tuple(teacher_base_topk_log_probs.shape)} != {tuple(expected)}"
        )
    if weighting not in {"uniform", "cosine"}:
        raise ValueError(f"unsupported ExpertAlign weighting {weighting!r}")
    if alignment_epsilon < 0 or norm_epsilon <= 0:
        raise ValueError("ExpertAlign epsilons must be non-negative/positive")

    student = student_topk_log_probs.detach().float()
    teachers = teacher_topk_log_probs.detach().float()
    base = teacher_base_topk_log_probs.detach().float()
    specialization = teachers - base.unsqueeze(0)
    teaching = teachers - student.unsqueeze(0)
    specialization_norm = specialization.square().sum(dim=-1).sqrt()
    teaching_norm = teaching.square().sum(dim=-1).sqrt()
    inner = (specialization * teaching).sum(dim=-1)
    retained = (
        (inner > alignment_epsilon)
        & (specialization_norm > norm_epsilon)
        & (teaching_norm > norm_epsilon)
    )
    cosine = inner / (specialization_norm * teaching_norm).clamp_min(norm_epsilon)
    cosine = torch.where(retained, cosine, torch.zeros_like(cosine))

    scores = retained.to(cosine.dtype) if weighting == "uniform" else cosine
    denominator = scores.sum(dim=0, keepdim=True)
    weights = scores / denominator.clamp_min(norm_epsilon)
    routed = (weights.unsqueeze(-1) * teachers).sum(dim=0)
    any_teacher = denominator.squeeze(0) > 0
    routed = torch.where(any_teacher.unsqueeze(-1), routed, student)
    return routed.to(teacher_topk_log_probs.dtype), weights, cosine, specialization_norm
