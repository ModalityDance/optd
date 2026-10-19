"""Check streamed path probabilities and gradients against dense vocabulary logits."""
import torch
from torch import nn
from verl.utils.transition_support import snapshot, path_readout

from unittest.mock import patch


def test_streamed_readout_matches_dense_values_and_gradients():
    # Exercise the differentiable student branch on CPU. The ordinary CPU
    # branch serves parked teacher labels and intentionally detaches them.
    with patch("verl.utils.jlens_distillation._use_host_readout", return_value=False):
        torch.manual_seed(71)
        hidden = torch.randn(1, 28, 3, 8, device="cpu", requires_grad=True)
        norm = nn.RMSNorm(8).to("cpu")
        head = nn.Linear(8, 192, bias=False).to("cpu")
        layers = [7, 13, 19]
        with torch.no_grad():
            ids, frozen = snapshot(hidden, layers, norm, head, 2)
        dense = torch.stack([head(norm(hidden[:, i])) if i != 27 else head(hidden[:, i])
                             for i in [*layers, 27]], 1).float().log_softmax(-1)
        expected_ids = torch.cat([dense[:, i].topk(32, -1).indices for i in range(4)], -1)
        assert torch.equal(ids, expected_ids)
        expected = dense.gather(-1, ids[:, None].expand(-1, 4, -1, -1))
        actual = path_readout(hidden, layers, ids, norm, head, 2)
        torch.testing.assert_close(frozen, expected.detach(), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        weight = torch.randn_like(actual)
        grad_a = torch.autograd.grad((actual * weight).sum(), hidden, retain_graph=True)[0]
        grad_b = torch.autograd.grad((expected * weight).sum(), hidden)[0]
        torch.testing.assert_close(grad_a, grad_b, atol=3e-5, rtol=3e-5)
        print("DENSE_READOUT_AND_GRADIENT_OK")
