"""Selection must use snapshot error, deduplicate, and backpropagate both endpoints."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("transition_support", Path(__file__).parents[1] / "verl/verl/utils/transition_support.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_selection_and_gradient():
    torch.manual_seed(42)
    full = torch.randn(1, 4, 2, 160).log_softmax(-1)
    teacher = torch.randn_like(full).log_softmax(-1)
    ids = torch.cat([full[:, i].topk(32, -1).indices for i in range(4)], -1)
    index = ids[:, None].expand(-1, 4, -1, -1)
    s, t = full.gather(-1, index), teacher.gather(-1, index)
    for mode in ("max", "min"):
        selected = m.select_indices(ids, s, t, mode)
        for pair in range(3):
            for token in range(2):
                union = set(ids[0, token, pair*32:pair*32+64].tolist())
                error = ((full[0, pair+1, token] - full[0, pair, token]) -
                         (teacher[0, pair+1, token] - teacher[0, pair, token])).abs()
                expected = sorted(union, key=lambda v: float(error[v]), reverse=mode == "max")[:16]
                actual = ids[0, token].gather(0, selected[0, pair, token]).tolist()
                assert set(actual) == set(expected) and len(set(actual)) == 16
        current = s.clone().requires_grad_()
        loss = m.losses(current, t, selected, torch.ones(1, 2)).sum()
        loss.backward()
        assert torch.isfinite(current.grad).all()
        assert (current.grad.abs().sum((0, 2, 3)) > 0).all()
        assert torch.allclose(current.grad.sum(1), torch.zeros_like(current.grad[:, 0]), atol=1e-6)


if __name__ == "__main__":
    test_selection_and_gradient()
    print("PAIR_SELECTION_AND_GRADIENT_OK")
