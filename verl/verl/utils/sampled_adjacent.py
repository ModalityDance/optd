"""Uniform, minibatch-synchronous sampling of adjacent decoder transitions."""
import torch


def adjacent_slots_for_batch(*, batch_size, global_minibatch_size, world_size,
                             num_pairs, seed, device=None):
    """Assign after DP balancing: rank-major rows, one common pair per update."""
    if (min(batch_size, global_minibatch_size, world_size, num_pairs) <= 0
            or batch_size % global_minibatch_size
            or global_minibatch_size % world_size):
        raise ValueError("adjacent sampling requires complete, equally sharded minibatches")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    updates = torch.randint(num_pairs, (batch_size // global_minibatch_size,), generator=generator)
    local = updates.repeat_interleave(global_minibatch_size // world_size)
    return local.repeat(world_size).to(device=device), updates


def sampled_adjacent_readout(hidden, slots, final_log_probs, candidate_ids, *,
                             final_norm, lm_head, token_chunk_size=32, temperature=1.0):
    """Read only the two selected endpoints; use true final logits for last block.

    HF's last hidden-state entry has already passed final normalization. Never
    read that entry with the raw intermediate lens (which would normalize twice).
    Earlier entries are decoder block outputs indexed from zero.
    """
    from verl.utils.jlens_distillation import layerwise_log_probs_on_policy_topk
    slots = slots.to(device=hidden.device, dtype=torch.long).reshape(-1)
    num_pairs = hidden.size(1) - 1
    if slots.numel() != hidden.size(0) or bool(((slots < 0) | (slots >= num_pairs)).any()):
        raise ValueError("one valid adjacent-pair index is required per sample")
    rows = torch.arange(hidden.size(0), device=hidden.device)
    kwargs = dict(final_norm=final_norm, lm_head=lm_head,
                  token_chunk_size=token_chunk_size, temperature=temperature)
    source = layerwise_log_probs_on_policy_topk(
        hidden[rows, slots].unsqueeze(1), candidate_ids, **kwargs)[:, 0]
    # Index-copy preserves the gradient of the chosen destination, including
    # the ordinary final-policy graph for the last pair.
    nonfinal = torch.nonzero(slots < num_pairs - 1, as_tuple=False).flatten()
    destination = final_log_probs.to(source.device)
    if nonfinal.numel():
        dest_hidden = hidden[nonfinal, slots[nonfinal] + 1].unsqueeze(1)
        dest = layerwise_log_probs_on_policy_topk(
            dest_hidden, candidate_ids.index_select(0, nonfinal.to(candidate_ids.device)),
            **kwargs)[:, 0]
        destination = destination.index_copy(0, nonfinal.to(destination.device), dest)
    return torch.stack((source, destination), dim=1)
