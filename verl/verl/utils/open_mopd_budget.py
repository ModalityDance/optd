"""Open-MOPD Eqs. (8)-(9), measured on one complete rollout batch."""
import torch

CODE_SOURCES = frozenset({"codecontests", "apps", "codeforces", "taco"})


@torch.no_grad()
def apply_budget(batch, alpha=1.0):
    mask = batch.batch["response_mask"].double()
    reward = batch.batch["token_level_scores"].double()
    if reward.ndim == 3:
        reward = reward.sum(-1)  # r_t = sum_v r_t(v), paper Eq. (4)
    code = torch.tensor([str(s) in CODE_SOURCES for s in batch.non_tensor_batch["data_source"]],
                        device=mask.device)
    masks = torch.stack([mask * (~code).unsqueeze(-1), mask * code.unsqueeze(-1)])
    counts = masks.sum((1, 2))
    if (counts <= 0).any():
        raise ValueError("Open-MOPD requires both domains in the complete rollout batch")
    magnitudes = (masks * reward.abs().unsqueeze(0)).sum((1, 2)) / counts
    shares = counts / counts.sum()
    reference = magnitudes.mean()
    factors = ((magnitudes / reference.clamp_min(1e-12)) ** alpha).clamp(0.05, 20)
    weights = 0.5 / shares * factors
    weights /= (weights * shares).sum()
    row_weights = weights[code.long()]
    advantages = batch.batch["advantages"]
    batch.batch["advantages"] = advantages * row_weights.to(advantages.dtype).reshape(
        -1, *([1] * (advantages.ndim - 1)))
    metrics = {}
    for i, domain in enumerate(("math", "code")):
        for name, value in (("tokens", counts[i]), ("magnitude", magnitudes[i]),
                            ("token_share", shares[i]), ("weight", weights[i]),
                            ("budget_share", weights[i] * shares[i])):
            metrics[f"open_mopd/{domain}_{name}"] = value.item()
    metrics["open_mopd/weighted_share_sum"] = (weights * shares).sum().item()
    return metrics
