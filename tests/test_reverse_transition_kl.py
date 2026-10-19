"""Numerical gradient contract for the actual training helper (CPU only)."""
import json
import torch
from verl.utils.jlens_distillation import masked_layerwise_transition_opd_policy_losses_from_log_probs as loss_fn


def check(dtype=torch.float64):
    torch.manual_seed(20261005)
    student = torch.randn(2, 4, 5, 16, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=dtype)
    diagnostics = {}
    surrogate = loss_fn(teacher[:, :-1], student[:, :-1], teacher[:, -1], student[:, -1],
                        mask, reverse_transition_kl=True, diagnostics=diagnostics).mean()
    sd = student[:, 1:] - student[:, :-1]
    td = (teacher[:, 1:] - teacher[:, :-1]).detach()
    logqs, logqt = sd.log_softmax(-1), td.log_softmax(-1)
    direct = (((logqs.exp() * (logqs - logqt)).sum(-1) * mask[:, None]).sum()
              / mask.sum() / 3)
    gs = torch.autograd.grad(surrogate, student, retain_graph=True)[0]
    gd = torch.autograd.grad(direct, student)[0]
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(gs, gd, rtol=tolerance, atol=tolerance)
    assert torch.autograd.grad(surrogate, teacher, allow_unused=True)[0] is None
    assert gs.permute(0, 2, 1, 3)[mask == 0].count_nonzero() == 0
    assert all(value == 1 for value in diagnostics.values())
    return dict(dtype=str(dtype), max_abs_gradient_error=(gs-gd).abs().max().item(),
                cosine=torch.nn.functional.cosine_similarity(gs.flatten(), gd.flatten(), dim=0).item(),
                direct_kl=direct.item(), diagnostics=diagnostics)


def test_gradient_matches_direct_reverse_kl():
    check()
    check(torch.float32)


def test_equal_increment_and_constant_shift_have_zero_gradient():
    torch.manual_seed(4)
    teacher = torch.randn(1, 4, 3, 8, dtype=torch.float64)
    student = (teacher + torch.randn(1, 4, 3, 1, dtype=torch.float64)).requires_grad_()
    loss = loss_fn(teacher[:, :-1], student[:, :-1], teacher[:, -1], student[:, -1],
                   torch.ones(1, 3), reverse_transition_kl=True).mean()
    grad = torch.autograd.grad(loss, student)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-14, rtol=0)


if __name__ == '__main__':
    print(json.dumps([check(), check(torch.float32)], indent=2))
