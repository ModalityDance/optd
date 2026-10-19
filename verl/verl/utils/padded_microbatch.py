"""Trim only padding shared by every row, without changing micro-batch membership."""
import torch


RESPONSE_FIELDS = frozenset({
    "responses", "response_mask", "old_log_probs", "advantages", "ref_log_prob",
    "rollout_is_weights", "student_top_k_ids", "student_top_k_log_probs",
    "teacher_on_student_log_probs", "union_top_k_ids", "union_top_k_log_probs",
})


def trim_common_padding(inputs):
    if "multi_modal_inputs" in inputs:
        raise ValueError("common-padding trim currently supports text-only inputs")
    result = dict(inputs)
    attention = inputs["attention_mask"].bool()
    response_width = inputs["responses"].shape[1]
    prompt_width = attention.shape[1] - response_width
    if prompt_width < 1 or inputs["position_ids"].ndim != 2:
        raise ValueError("expected a text prompt and two-dimensional position ids")
    used = attention.any(0)
    prompt_used = used[:prompt_width].nonzero().flatten()
    response_used = (used[prompt_width:] | inputs["response_mask"].bool().any(0)).nonzero().flatten()
    # Keep a prompt token for predicting response[0], even for an empty prompt.
    left = min(int(prompt_used[0]) if prompt_used.numel() else prompt_width - 1, prompt_width - 1)
    width = max(int(response_used[-1]) + 1 if response_used.numel() else 0, 1)
    for key in ("input_ids", "attention_mask", "position_ids"):
        result[key] = inputs[key][:, left:prompt_width + width].contiguous()
    for key in RESPONSE_FIELDS & inputs.keys():
        value = inputs[key]
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] != response_width:
            raise ValueError(f"unexpected response-aligned field {key}")
        result[key] = value[:, :width].contiguous()
    # Compact last-k teacher labels and frozen support are already indexed by
    # valid response position, not by the padded response axis. Leave them intact.
    return result
