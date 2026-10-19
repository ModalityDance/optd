import torch

from verl.utils.jlens_distillation import (
    masked_layerwise_transition_opd_policy_losses_from_log_probs,
)


def test_transition_opd_zero_when_student_matches_teacher_path():
    torch.manual_seed(0)
    teacher_layers = torch.log_softmax(torch.randn(2, 3, 5, 4), dim=-1)
    student_layers = teacher_layers.clone().requires_grad_()
    teacher_final = torch.log_softmax(torch.randn(2, 5, 4), dim=-1)
    student_final = teacher_final.clone().requires_grad_()
    mask = torch.ones(2, 5)

    losses = masked_layerwise_transition_opd_policy_losses_from_log_probs(
        teacher_layers, student_layers, teacher_final, student_final, mask
    )

    assert losses.shape == (3,)
    assert torch.allclose(losses, torch.zeros_like(losses), atol=1e-6)
    losses.sum().backward()
    assert student_layers.grad is not None
    assert student_final.grad is not None


def test_transition_opd_requires_common_support_shapes():
    args = [
        torch.zeros(1, 3, 2, 4),
        torch.zeros(1, 3, 2, 4),
        torch.zeros(1, 2, 5),
        torch.zeros(1, 2, 4),
        torch.ones(1, 2),
    ]
    try:
        masked_layerwise_transition_opd_policy_losses_from_log_probs(*args)
    except ValueError as exc:
        assert "final transition" in str(exc)
    else:
        raise AssertionError("mismatched transition support should fail")
