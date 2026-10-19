"""Shared runtime utilities for JLens-based on-policy alignment.

Three objectives share Jacobian transport into a common space:

* ``opd`` / J-OPD: Jh → Norm + LM-head → only_stu KL-reward → score-function OPD
  on the final-policy top-k support (fair pair with LayerwiseOPD+OPD).
  **JOPD-L** is the single-layer special case (``layers=last`` / deepest fitted
  ``J_w``): same OPD top-k recipe with one extra frozen linear map before the head.
* ``kl`` / JAlign-Softmax: final Norm + LM-head → ``T^2 KL`` on distributions
  (ablation / DirectSoftmax-style; not the formal OPD pair).
* ``mse`` / JAlign-MSE: L2-normalized MSE on the transported vectors
  (pairs with OPRD representation MSE; no Norm / unembedding).

Student transport modes:

* ``frozen``: apply a fitted ``lens.pt`` matrix ``J h``.
* ``stochastic_low_rank``: per-step Hutchinson estimate of the *current*
  student map ``h_m → h_L`` via finite-difference JVPs (idea.md §3).

Same-tokenizer full-vocabulary logits are supported for the KL path.  A
versioned vocabulary subset remains available for debugging or constrained
cross-tokenizer settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def parse_jalign_layers(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse explicit post-block layer IDs and reject ambiguous mappings.

    The sentinel ``last`` is not resolved here; use ``resolve_jalign_layers`` with a
    lens checkpoint when the caller wants the deepest fitted JLens layer (JOPD-L).
    """
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "last":
            raise ValueError(
                "layer sentinel 'last' requires resolve_jalign_layers(..., lens_checkpoint=...); "
                "parse_jalign_layers only accepts explicit integer IDs"
            )
        try:
            layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"invalid JAlign layer list {value!r}") from exc
    else:
        layers = tuple(int(layer) for layer in value)
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError(f"JAlign layers must be distinct non-negative integers, got {layers}")
    return layers


def peek_jlens_layers(lens_path: str | Path) -> tuple[int, ...]:
    """Return sorted source-layer IDs stored in a ``lens.pt`` checkpoint."""
    resolved_lens = _resolve_lens_path(lens_path)
    if not resolved_lens.is_file():
        raise FileNotFoundError(f"JLens checkpoint does not exist: {resolved_lens}")
    lens_payload = torch.load(resolved_lens, map_location="cpu", weights_only=True)
    if not isinstance(lens_payload, dict) or "J" not in lens_payload:
        raise ValueError(f"{resolved_lens} is not a JLens checkpoint")
    return tuple(sorted(int(layer) for layer in lens_payload["J"].keys()))


def resolve_jalign_layers(
    value: str | Sequence[int],
    *,
    lens_checkpoint: str | Path | None = None,
) -> tuple[int, ...]:
    """Parse layer IDs; ``last`` / ``jopd_l`` selects the deepest fitted JLens layer.

    **JOPD-L** (Last-layer Jacobian OPD): only the deepest fitted ``J_w`` is used.
    Relative to final-policy OPD top-k, the auxiliary is the same only_stu
    KL-reward → score-function OPD, with one extra frozen linear map
    ``h ↦ J_w h`` before Norm+LM-head on that layer.
    """
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"last", "jopd_l", "jopd-l"}:
            if lens_checkpoint is None:
                raise ValueError(f"JAlign layers={value!r} requires a lens checkpoint to resolve 'last'")
            available = peek_jlens_layers(lens_checkpoint)
            if not available:
                raise ValueError(f"no layers in JLens checkpoint {lens_checkpoint}")
            return (available[-1],)
    return parse_jalign_layers(value)


def validate_jalign_loss_type(value: str) -> str:
    """Validate the JAlign objective, including temporal J-Flow OPD."""
    normalized = str(value).strip().lower()
    if normalized not in {"opd", "flow_opd", "kl", "mse"}:
        raise ValueError(
            f"jalign_loss_type must be 'opd', 'flow_opd', 'kl', or 'mse', got {value!r}"
        )
    return normalized


def validate_jalign_student_transport(value: str) -> str:
    """Accept fitted-lens modes or per-step stochastic FD-JVP.

    * ``frozen``: keep the initial ``lens.pt`` for the whole run.
    * ``periodic_refresh``: same transport as frozen, but trainer re-fits the
      student lens every ``jalign_refresh_interval`` steps (idea.md §1).
    * ``stochastic_low_rank``: per-step Hutchinson FD-JVP (idea.md §3).
    * ``conditional_residual``: a frozen-base low-rank adapter trained only
      with its detached transport MSE.
    * ``identity_adapter``: the same zero-residual initialization, jointly
      trained by JOPD and its transport MSE constraint.
    """
    normalized = str(value).strip().lower()
    allowed = {"frozen", "periodic_refresh", "stochastic_low_rank", "conditional_residual", "identity_adapter"}
    if normalized not in allowed:
        raise ValueError(f"jalign_student_transport must be one of {sorted(allowed)}, got {value!r}")
    return normalized


def validate_jalign_gradient_routing(value: str) -> str:
    """Select exact JLens backward or identity-credit surrogate backward."""
    normalized = str(value).strip().lower()
    allowed = {"jacobian", "identity_credit"}
    if normalized not in allowed:
        raise ValueError(f"jalign_gradient_routing must be one of {sorted(allowed)}, got {value!r}")
    return normalized


def jalign_uses_fitted_lens(transport: str) -> bool:
    """True when student transport applies a stored ``J`` matrix (not FD-JVP)."""
    return validate_jalign_student_transport(transport) in {
        "frozen", "periodic_refresh", "conditional_residual", "identity_adapter"
    }


def validate_jalign_stoch_probe(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"rademacher", "gaussian"}:
        raise ValueError(f"jalign_stoch_probe must be 'rademacher' or 'gaussian', got {value!r}")
    return normalized


def sample_jalign_stoch_probe(
    d_model: int,
    *,
    probe: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw a unit-scale probe with ``E[v v^T] = I``."""
    probe = validate_jalign_stoch_probe(probe)
    if probe == "rademacher":
        bits = torch.randint(0, 2, (d_model,), device=device, generator=generator, dtype=torch.long)
        return bits.mul(2).sub(1).to(dtype=dtype)
    return torch.randn(d_model, device=device, dtype=dtype, generator=generator)


def hutchinson_transport_from_jvps(
    residuals: Tensor,
    probes: Sequence[Tensor],
    jvps: Sequence[Tensor],
) -> Tensor:
    """Apply ``(1/r) Σ sg(Jv_i)(v_i^T h)`` for a single layer ``[B,T,D]``.

    ``jvps`` are treated as stop-grad; gradients flow through ``residuals`` via
    the scalar projections ``v_i^T h``.
    """
    if residuals.dim() != 3:
        raise ValueError(f"expected residuals [B,T,D], got {tuple(residuals.shape)}")
    if len(probes) == 0 or len(probes) != len(jvps):
        raise ValueError("probes and jvps must be non-empty and aligned")
    acc = None
    for probe, jvp in zip(probes, jvps, strict=True):
        if probe.numel() != residuals.size(-1) or jvp.shape != residuals.shape:
            raise ValueError(
                f"probe/jvp shape mismatch for residuals {tuple(residuals.shape)}: "
                f"probe={tuple(probe.shape)}, jvp={tuple(jvp.shape)}"
            )
        scalar = (residuals * probe.reshape(1, 1, -1).to(device=residuals.device, dtype=residuals.dtype)).sum(
            dim=-1, keepdim=True
        )
        term = jvp.detach().to(dtype=residuals.dtype) * scalar
        acc = term if acc is None else acc + term
    return acc / float(len(probes))


def resolve_decoder_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the HF decoder block list (``model.model.layers`` / ``transformer.h``)."""
    base = getattr(model, "module", model)
    backbone = getattr(base, "model", None)
    blocks = getattr(backbone, "layers", None) if backbone is not None else None
    if blocks is None:
        transformer = getattr(base, "transformer", None)
        blocks = getattr(transformer, "h", None) if transformer is not None else None
    if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
        raise ValueError("stochastic JAlign requires a decoder ModuleList at model.layers or transformer.h")
    return blocks


def _hook_add_probe(probe: Tensor, eps: float):
    """Forward hook that adds ``eps * probe`` to a block's residual-stream output."""

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            pert = hidden + eps * probe.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            return (pert,) + output[1:]
        return output + eps * probe.to(device=output.device, dtype=output.dtype).view(1, 1, -1)

    return hook


def _compact_response_hidden(
    full_hidden: Tensor,
    response_mask: Tensor,
    *,
    positions: str,
    last_k: int,
    first_k: int,
) -> Tensor:
    from verl.utils.rep_distillation import (
        compact_response_repr_by_positions,
        extract_response_hidden_states,
    )

    response_hidden = extract_response_hidden_states(full_hidden.float(), response_mask)
    compact_hidden, _ = compact_response_repr_by_positions(
        response_hidden, response_mask, positions, last_k=last_k, first_k=first_k
    )
    return compact_hidden


def estimate_layer_jvp_fd(
    model: nn.Module,
    *,
    layer_id: int,
    probe: Tensor,
    eps: float,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    response_mask: Tensor,
    base_final_hidden: Tensor,
    positions: str,
    last_k: int,
    first_k: int,
) -> Tensor:
    """Finite-difference JVP of final response residuals w.r.t. layer-``layer_id`` residual.

    Returns stop-grad ``[B, T, D]`` matching ``base_final_hidden``.
    """
    if eps <= 0:
        raise ValueError(f"jalign_stoch_eps must be positive, got {eps}")
    blocks = resolve_decoder_blocks(model)
    if layer_id < 0 or layer_id >= len(blocks):
        raise ValueError(f"layer_id {layer_id} out of range for {len(blocks)} decoder blocks")

    from verl.utils.device import get_device_name

    handle = blocks[layer_id].register_forward_hook(_hook_add_probe(probe, eps))
    try:
        with torch.no_grad():
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
            pert_final = _compact_response_hidden(
                output.hidden_states[-1],
                response_mask,
                positions=positions,
                last_k=last_k,
                first_k=first_k,
            )
    finally:
        handle.remove()
    return ((pert_final - base_final_hidden) / eps).detach()


def stochastic_jlens_transported_residuals(
    model: nn.Module,
    *,
    source_layers: Sequence[int],
    source_residuals: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    response_mask: Tensor,
    positions: str,
    last_k: int,
    first_k: int,
    rank: int,
    eps: float = 1e-3,
    probe: str = "rademacher",
    use_remove_padding: bool = False,
    use_ulysses_sp: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Per-step stochastic low-rank transport of ``[B,L,T,D]`` student residuals.

    Estimates ``J_m v`` with finite-difference re-forwards (stop-grad), then
    applies Hutchinson's identity so gradients enter only through ``source_residuals``.
    """
    if use_remove_padding or use_ulysses_sp:
        raise NotImplementedError(
            "stochastic_low_rank JAlign currently requires use_remove_padding=False and no Ulysses SP"
        )
    layers = parse_jalign_layers(source_layers)
    if source_residuals.dim() != 4 or source_residuals.size(1) != len(layers):
        raise ValueError(
            f"source_residuals must be [B,L,T,D] with L={len(layers)}, got {tuple(source_residuals.shape)}"
        )
    if rank <= 0:
        raise ValueError(f"jalign_stoch_rank must be positive, got {rank}")
    probe = validate_jalign_stoch_probe(probe)

    from verl.utils.device import get_device_name

    with torch.no_grad():
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            base_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
        base_final = _compact_response_hidden(
            base_out.hidden_states[-1],
            response_mask,
            positions=positions,
            last_k=last_k,
            first_k=first_k,
        )
    if base_final.shape[:2] != source_residuals.shape[:1] + source_residuals.shape[2:3]:
        raise ValueError(
            f"compacted final hidden {tuple(base_final.shape)} incompatible with "
            f"source residuals {tuple(source_residuals.shape)}"
        )

    d_model = source_residuals.size(-1)
    layer_outputs = []
    for offset, layer_id in enumerate(layers):
        layer_h = source_residuals[:, offset]
        probes: list[Tensor] = []
        jvps: list[Tensor] = []
        for _ in range(rank):
            v = sample_jalign_stoch_probe(
                d_model,
                probe=probe,
                device=layer_h.device,
                dtype=torch.float32,
                generator=generator,
            )
            jv = estimate_layer_jvp_fd(
                model,
                layer_id=layer_id,
                probe=v,
                eps=eps,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_mask=response_mask,
                base_final_hidden=base_final,
                positions=positions,
                last_k=last_k,
                first_k=first_k,
            )
            probes.append(v)
            jvps.append(jv)
        layer_outputs.append(hutchinson_transport_from_jvps(layer_h, probes, jvps))
    return torch.stack(layer_outputs, dim=1)


def readout_selected_logits_from_residuals(
    residuals: Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_ids: Tensor | None = None,
) -> Tensor:
    """Apply final Norm + LM-head to transported ``[B,L,T,D]`` residuals."""
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if not hasattr(lm_head, "weight"):
        raise ValueError("JAlign requires an lm_head with a weight tensor")
    outputs = []
    for offset in range(residuals.size(1)):
        normalized = final_norm(residuals[:, offset])
        weights = lm_head.weight if token_ids is None else lm_head.weight.index_select(
            0, token_ids.to(residuals.device, non_blocking=True)
        )
        weights = weights.to(normalized.dtype)
        bias = getattr(lm_head, "bias", None)
        if bias is not None and token_ids is not None:
            bias = bias.index_select(0, token_ids.to(residuals.device, non_blocking=True))
        if bias is not None:
            bias = bias.to(normalized.dtype)
        outputs.append(F.linear(normalized, weights, bias).float())
    return torch.stack(outputs, dim=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_selected_vocab_ids(mapping_path: str | Path | None) -> Tensor | None:
    """Load an optional vocabulary subset; ``None`` means the complete LM-head vocab."""
    if mapping_path is None or str(mapping_path).strip().lower() in {"", "none", "null"}:
        return None
    mapping = Path(mapping_path).expanduser().resolve()
    mapping_payload = _load_json(mapping)
    if mapping_payload.get("mapping_type") != "identical_tokenizer_ids":
        raise ValueError("alignment supports only identical_tokenizer_ids mappings")
    entries = mapping_payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("alignment mapping contains no entries")
    token_ids = [entry.get("token_id") for entry in entries]
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids):
        raise ValueError("alignment mapping has invalid token IDs")
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("alignment mapping has duplicate token IDs")
    return torch.tensor(token_ids, dtype=torch.long)


def _resolve_lens_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate / "lens.pt" if candidate.is_dir() else candidate


@dataclass
class JLensArtifacts:
    """Validated Jacobian matrices and shared vocabulary IDs kept on CPU."""

    jacobians: dict[int, Tensor]
    token_ids: Tensor | None
    layers: tuple[int, ...]
    d_model: int
    mapping_path: Path | None

    @classmethod
    def load(
        cls,
        lens_path: str | Path,
        mapping_path: str | Path | None,
        *,
        expected_layers: Sequence[int] | None = None,
    ) -> "JLensArtifacts":
        resolved_lens = _resolve_lens_path(lens_path)
        if not resolved_lens.is_file():
            raise FileNotFoundError(f"JLens checkpoint does not exist: {resolved_lens}")
        lens_payload = torch.load(resolved_lens, map_location="cpu", weights_only=True)
        if not isinstance(lens_payload, dict) or "J" not in lens_payload:
            raise ValueError(f"{resolved_lens} is not a JLens checkpoint")
        d_model = int(lens_payload["d_model"])
        jacobians_all = {int(layer): tensor.float().contiguous() for layer, tensor in lens_payload["J"].items()}
        available = tuple(sorted(jacobians_all))
        if not available or any(matrix.shape != (d_model, d_model) for matrix in jacobians_all.values()):
            raise ValueError("JLens checkpoint has invalid Jacobian shapes")
        if expected_layers is None:
            layers = available
            jacobians = jacobians_all
        else:
            layers = tuple(int(layer) for layer in expected_layers)
            missing = [layer for layer in layers if layer not in jacobians_all]
            if missing:
                raise ValueError(
                    f"JLens checkpoint layers {available} missing requested layers {missing}"
                )
            if len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
                raise ValueError(f"JAlign layers must be distinct non-negative integers, got {layers}")
            jacobians = {layer: jacobians_all[layer] for layer in layers}

        mapping = None if mapping_path is None or str(mapping_path).strip().lower() in {"", "none", "null"} else Path(mapping_path).expanduser().resolve()
        token_ids = load_selected_vocab_ids(mapping)

        manifest_path = resolved_lens.parent / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            expected_checksum = manifest.get("lens_file_sha256")
            if expected_checksum and expected_checksum != _sha256_file(resolved_lens):
                raise ValueError(f"JLens checksum mismatch for {resolved_lens}")
            if manifest.get("hidden_size") not in (None, d_model):
                raise ValueError("JLens manifest hidden size does not match checkpoint")

        return cls(
            jacobians=jacobians,
            token_ids=token_ids,
            layers=layers,
            d_model=d_model,
            mapping_path=mapping,
        )


@dataclass
class FrozenAffineReadoutArtifacts:
    """Validated layer-specific affine translators kept on CPU."""

    weights: dict[int, Tensor]
    biases: dict[int, Tensor]
    layers: tuple[int, ...]
    d_model: int
    _device_cache: dict[tuple[str, torch.dtype], tuple[Tensor, Tensor]]

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        expected_layers: Sequence[int],
    ) -> "FrozenAffineReadoutArtifacts":
        root = Path(checkpoint_dir).expanduser().resolve()
        layers = tuple(int(layer) for layer in expected_layers)
        if not root.is_dir():
            raise FileNotFoundError(f"affine readout directory does not exist: {root}")
        if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
            raise ValueError(f"affine readout layers must be distinct non-negative integers, got {layers}")
        weights: dict[int, Tensor] = {}
        biases: dict[int, Tensor] = {}
        d_model = -1
        for layer in layers:
            path = root / f"layer_{layer}_affine.pt"
            if not path.is_file():
                raise FileNotFoundError(f"affine readout checkpoint does not exist: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or int(payload.get("layer", -1)) != layer:
                raise ValueError(f"{path} does not identify decoder block {layer}")
            state = payload.get("state_dict")
            if not isinstance(state, dict):
                raise ValueError(f"{path} has no affine state_dict")
            weight = state.get("linear.weight")
            bias = state.get("linear.bias")
            width = int(payload.get("width", -1))
            if not isinstance(weight, Tensor) or weight.shape != (width, width):
                raise ValueError(f"{path} has invalid affine weight shape")
            if not isinstance(bias, Tensor) or bias.shape != (width,):
                raise ValueError(f"{path} has invalid affine bias shape")
            if d_model not in {-1, width}:
                raise ValueError("affine readout checkpoints have inconsistent hidden widths")
            d_model = width
            weights[layer] = weight.float().contiguous()
            biases[layer] = bias.float().contiguous()
        return cls(
            weights=weights,
            biases=biases,
            layers=layers,
            d_model=d_model,
            _device_cache={},
        )

    def tensors_for(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        key = (str(reference.device), reference.dtype)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = (
                torch.stack([self.weights[layer] for layer in self.layers]).to(
                    device=reference.device, dtype=reference.dtype
                ),
                torch.stack([self.biases[layer] for layer in self.layers]).to(
                    device=reference.device, dtype=reference.dtype
                ),
            )
            self._device_cache[key] = cached
        return cached


def frozen_affine_readout_residuals(
    residuals: Tensor,
    *,
    artifacts: FrozenAffineReadoutArtifacts,
) -> Tensor:
    """Apply frozen per-layer affine translators without detaching residuals."""
    if residuals.ndim != 4:
        raise ValueError(f"affine readout expects [B,L,T,D], got {tuple(residuals.shape)}")
    if residuals.size(1) != len(artifacts.layers) or residuals.size(-1) != artifacts.d_model:
        raise ValueError(
            f"residual shape {tuple(residuals.shape)} is incompatible with "
            f"layers={artifacts.layers}, d_model={artifacts.d_model}"
        )
    weights, biases = artifacts.tensors_for(residuals)
    return torch.einsum("bltd,led->blte", residuals, weights) + biases[None, :, None, :]


class ConditionalResidualJLens(nn.Module):
    """Frozen JLens plus one rank-r residual per selected source layer."""

    def __init__(self, artifacts: JLensArtifacts, rank: int = 1):
        super().__init__()
        if rank <= 0:
            raise ValueError("conditional JLens rank must be positive")
        self.layers = artifacts.layers
        self.rank = rank
        d = artifacts.d_model
        self.register_buffer("base", torch.stack([artifacts.jacobians[layer] for layer in self.layers]))
        self.left = nn.Parameter(torch.zeros(len(self.layers), d, rank))
        self.right = nn.Parameter(torch.randn(len(self.layers), rank, d) * 0.01)

    def forward(self, residuals: Tensor, *, detach_params: bool = False) -> Tensor:
        if residuals.dim() != 4 or residuals.size(1) != len(self.layers):
            raise ValueError("conditional JLens residual shape does not match selected layers")
        left, right = self.left, self.right
        if detach_params:
            left, right = left.detach(), right.detach()
        base = self.base.to(device=residuals.device, dtype=residuals.dtype)
        transported = torch.einsum("bltd,lde->blte", residuals, base.transpose(-1, -2))
        factors = torch.einsum("bltd,lrd->bltr", residuals.float(), right.to(residuals.device))
        correction = torch.einsum("bltr,ldr->bltd", factors, left.to(residuals.device))
        return transported + correction.to(residuals.dtype)


def resolve_final_norm_and_lm_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Find the final norm and unembedding for decoder-only Hugging Face models."""
    base = getattr(model, "module", model)
    backbone = getattr(base, "model", None)
    norm = getattr(backbone, "norm", None)
    if norm is None:
        transformer = getattr(base, "transformer", None)
        norm = getattr(transformer, "ln_f", None)
    lm_head = getattr(base, "lm_head", None)
    if lm_head is None and hasattr(base, "get_output_embeddings"):
        lm_head = base.get_output_embeddings()
    if not isinstance(norm, nn.Module) or not isinstance(lm_head, nn.Module):
        raise ValueError(
            "JAlign currently requires a decoder model exposing model.norm/transformer.ln_f and lm_head"
        )
    return norm, lm_head


def jlens_transported_residuals(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
) -> Tensor:
    """Apply per-layer Jacobian transport: ``\\tilde h = J h``.

    Returns ``[B, L, T, D]`` in the same dtype/device as ``residuals``.  Gradients
    flow through ``residuals`` (student) while ``J`` stays a frozen buffer.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if residuals.size(1) != len(artifacts.layers) or residuals.size(-1) != artifacts.d_model:
        raise ValueError(
            f"residual shape {tuple(residuals.shape)} is incompatible with layers={artifacts.layers}, "
            f"d_model={artifacts.d_model}"
        )
    outputs = []
    for offset, layer in enumerate(artifacts.layers):
        jacobian = artifacts.jacobians[layer].to(device=residuals.device, dtype=residuals.dtype)
        outputs.append(residuals[:, offset] @ jacobian.T)
    return torch.stack(outputs, dim=1)


def jlens_identity_credit_residuals(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
) -> Tensor:
    """Use ``J h`` in forward while sending identity credit back to ``h``.

    This separates the JLens supervision target from its optimizer transport:
    readouts observe the fitted-lens value, but ``d output / d residuals = I``.
    The fitted Jacobian remains a non-trainable, periodically replaceable
    artifact and no autograd graph is retained for its matrix multiplication.
    """
    with torch.no_grad():
        transported = jlens_transported_residuals(residuals, artifacts=artifacts)
    return transported + (residuals - residuals.detach())


def jacobian_backward_identity_residuals(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
) -> Tensor:
    """Keep raw residuals in forward while routing their gradients through J.

    A Jacobian describes ``delta h_final = J delta h_source``, rather than a
    decoder for the absolute source state.  This straight-through construction
    therefore preserves LayerwiseOPD's raw intermediate readout exactly and
    changes only its backward path: ``d loss / d h = J.T d loss / d output``.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if residuals.size(1) != len(artifacts.layers) or residuals.size(-1) != artifacts.d_model:
        raise ValueError(
            f"residual shape {tuple(residuals.shape)} is incompatible with layers={artifacts.layers}, "
            f"d_model={artifacts.d_model}"
        )
    jacobians = torch.stack(
        [artifacts.jacobians[layer] for layer in artifacts.layers]
    ).to(device=residuals.device, dtype=residuals.dtype)
    delta = residuals - residuals.detach()
    return residuals.detach() + torch.einsum("bltd,lde->blte", delta, jacobians.transpose(-1, -2))


def resolve_layerwise_opd_gradient_routing(
    value: str,
    *,
    legacy_frozen_jacobian_backward: bool = False,
) -> str:
    """Resolve raw, frozen-JLens, and exact online-JCR gradient routing.

    The legacy boolean remains a compatibility alias for ``frozen_jlens`` so
    existing launchers and live runs cannot silently change semantics.
    """
    normalized = str(value).strip().lower()
    allowed = {
        "raw",
        "frozen_jlens",
        "online_jcr",
        "online_jcr_norm",
        "online_jcr_activation",
    }
    if normalized not in allowed:
        raise ValueError(
            f"layerwise_opd_gradient_routing must be one of {sorted(allowed)}, got {value!r}"
        )
    if legacy_frozen_jacobian_backward:
        if normalized in {"online_jcr", "online_jcr_norm", "online_jcr_activation"}:
            raise ValueError(
                f"{normalized} conflicts with legacy layerwise_opd_jacobian_backward=True"
            )
        return "frozen_jlens"
    return normalized


def validate_layerwise_opd_sample_one_layer(
    enabled: bool,
    *,
    gradient_routing: str,
    aggregation: str,
) -> None:
    """Validate the deliberately narrow sampled-one-layer raw OPD mode."""
    if not enabled:
        return
    if gradient_routing != "raw":
        raise ValueError(
            "sample-one-layer LayerwiseOPD requires layerwise_opd_gradient_routing=raw"
        )
    if aggregation != "legacy_additive":
        raise ValueError(
            "sample-one-layer LayerwiseOPD requires layerwise_opd_aggregation=legacy_additive"
        )


def layerwise_opd_round_robin_slot(cursor: int, num_layers: int) -> int:
    """Return the deterministic layer slot for one PPO minibatch."""
    if cursor < 0:
        raise ValueError("LayerwiseOPD round-robin cursor must be non-negative")
    if num_layers <= 0:
        raise ValueError("LayerwiseOPD round-robin requires at least one layer")
    return cursor % num_layers


def layerwise_opd_round_robin_slots_for_batch(
    *,
    batch_size: int,
    ppo_mini_batch_size: int,
    num_layers: int,
    start_cursor: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Assign one round-robin layer slot to every PPO minibatch in a batch."""
    if batch_size <= 0:
        raise ValueError("sample-one-layer batch size must be positive")
    if ppo_mini_batch_size <= 0 or batch_size % ppo_mini_batch_size != 0:
        raise ValueError(
            f"batch size {batch_size} must be divisible by PPO minibatch size {ppo_mini_batch_size}"
        )
    num_minibatches = batch_size // ppo_mini_batch_size
    minibatch_slots = torch.tensor(
        [
            layerwise_opd_round_robin_slot(start_cursor + offset, num_layers)
            for offset in range(num_minibatches)
        ],
        device=device,
        dtype=torch.long,
    )
    return minibatch_slots.repeat_interleave(ppo_mini_batch_size)


def online_jacobian_credit_residuals(
    source_residuals: Tensor,
    *,
    final_pre_norm_residual: Tensor,
    gradient_scale: Tensor | float = 1.0,
) -> Tensor:
    """Raw LayerOPD forward with exact current-input Jacobian credit routing.

    ``source_residuals`` contains one sampled intermediate layer.  The returned
    value is exactly that raw residual in forward.  In backward its cotangent
    enters the current actor's final pre-norm residual, so ordinary autograd
    computes the exact input-conditioned suffix VJP in the same FSDP backward
    pass.  No Jacobian is materialized and no gradient of a Jacobian is taken.
    """
    if source_residuals.dim() != 4 or source_residuals.size(1) != 1:
        raise ValueError(
            "online JCR expects one sampled source layer with shape [B,1,T,D]"
        )
    if final_pre_norm_residual.shape != (
        source_residuals.size(0),
        source_residuals.size(2),
        source_residuals.size(3),
    ):
        raise ValueError(
            "online JCR final residual must have shape [B,T,D] matching the source"
        )
    scale = torch.as_tensor(
        gradient_scale,
        device=source_residuals.device,
        dtype=source_residuals.dtype,
    )
    if scale.numel() != 1:
        raise ValueError("online JCR gradient scale must be scalar")
    target = final_pre_norm_residual.unsqueeze(1).to(source_residuals.dtype)
    return source_residuals.detach() + scale * (target - target.detach())


def sampled_online_jacobian_credit_residuals(
    source_residuals: Tensor,
    *,
    final_pre_norm_residual: Tensor,
    sampled_layer_slot: int,
    gradient_scale: Tensor | float = 1.0,
) -> Tensor:
    """Route one layer through Online-JCR while keeping other layers raw.

    All configured intermediate residuals remain present in the forward loss.
    Only the sampled layer replaces its direct backward path with the current
    actor suffix VJP; every non-sampled layer retains ordinary Layerwise-OPD
    gradients. The combined objective therefore still needs one backward.
    """
    if source_residuals.dim() != 4:
        raise ValueError(
            f"expected source residuals [B,L,T,D], got {tuple(source_residuals.shape)}"
        )
    if not 0 <= sampled_layer_slot < source_residuals.size(1):
        raise ValueError(
            f"sampled layer slot {sampled_layer_slot} is outside "
            f"[0, {source_residuals.size(1)})"
        )

    outputs = []
    for layer_slot in range(source_residuals.size(1)):
        residual = source_residuals[:, layer_slot : layer_slot + 1]
        if layer_slot == sampled_layer_slot:
            residual = online_jacobian_credit_residuals(
                residual,
                final_pre_norm_residual=final_pre_norm_residual,
                gradient_scale=gradient_scale,
            )
        outputs.append(residual)
    return torch.cat(outputs, dim=1)


def activation_only_online_jcr_layerwise_loss(
    per_layer_losses: Tensor,
    *,
    sampled_source_residual: Tensor,
    routed_cotangent: Tensor,
    sampled_layer_slot: int,
) -> Tensor:
    """Inject a detached suffix VJP at one layer while other losses stay raw.

    The returned scalar has the same forward value as ``per_layer_losses.mean()``.
    In backward, non-sampled losses remain ordinary Layerwise-OPD losses and the
    sampled loss injects the precomputed ``J.T @ u`` only at its source residual.
    The prepass that computes ``routed_cotangent`` uses a separate FSDP forward,
    so this formal update graph retains its untouched FSDP backward hooks.
    """
    if per_layer_losses.dim() != 1 or per_layer_losses.numel() == 0:
        raise ValueError("per-layer losses must be a non-empty rank-1 tensor")
    if not 0 <= sampled_layer_slot < per_layer_losses.numel():
        raise ValueError(
            f"sampled layer slot {sampled_layer_slot} is outside "
            f"[0, {per_layer_losses.numel()})"
        )
    if not sampled_source_residual.requires_grad:
        raise ValueError("sampled source residual must require gradients")
    if routed_cotangent.shape != sampled_source_residual.shape:
        raise ValueError(
            "activation-only Online-JCR routed cotangent must match the source "
            f"shape, got {tuple(routed_cotangent.shape)} and "
            f"{tuple(sampled_source_residual.shape)}"
        )

    sampled_loss = per_layer_losses[sampled_layer_slot]
    credit = (
        sampled_source_residual.float() * routed_cotangent.detach().float()
    ).sum()
    routed_sampled_loss = sampled_loss.detach() + credit - credit.detach()
    routed_losses = torch.cat(
        (
            per_layer_losses[:sampled_layer_slot],
            routed_sampled_loss.unsqueeze(0),
            per_layer_losses[sampled_layer_slot + 1 :],
        )
    )
    return routed_losses.mean()


def jlens_selected_logits(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
    final_norm: nn.Module,
    lm_head: nn.Module,
) -> Tensor:
    """Read selected-vocabulary J-logits from ``[B, L, T, D]`` residuals.

    Jacobian transport and the final norm retain gradients when ``residuals``
    belongs to the student actor. The final projection is restricted to the
    configured vocabulary before logits are returned.
    """
    transported = jlens_transported_residuals(residuals, artifacts=artifacts)
    return readout_selected_logits_from_residuals(
        transported,
        final_norm=final_norm,
        lm_head=lm_head,
        token_ids=artifacts.token_ids,
    )


def direct_selected_logits(
    residuals: Tensor,
    *,
    token_ids: Tensor | None,
    final_norm: nn.Module,
    lm_head: nn.Module,
) -> Tensor:
    """Read selected-vocabulary logits directly from intermediate residuals.

    This is the no-JLens control: it deliberately applies the ordinary final
    norm and unembedding to each selected intermediate activation, without
    Jacobian transport.  It returns ``[B, L, T, V_selected]``.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if not hasattr(lm_head, "weight"):
        raise ValueError("direct softmax alignment requires an lm_head weight")
    normalized = final_norm(residuals)
    weights = lm_head.weight if token_ids is None else lm_head.weight.index_select(
        0, token_ids.to(residuals.device, non_blocking=True)
    )
    weights = weights.to(normalized.dtype)
    bias = getattr(lm_head, "bias", None)
    if bias is not None and token_ids is not None:
        bias = bias.index_select(0, token_ids.to(residuals.device, non_blocking=True))
    if bias is not None:
        bias = bias.to(normalized.dtype)
    return F.linear(normalized, weights, bias).float()


def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_fsdp_wrapped_module", module)


def _rms_norm_weight(final_norm: nn.Module) -> Tensor | None:
    for candidate in (_unwrap_module(final_norm), final_norm):
        weight = getattr(candidate, "weight", None)
        if isinstance(weight, Tensor):
            return weight
    return None


def _rms_norm_device(final_norm: nn.Module, fallback: torch.device) -> torch.device:
    """Device of the RMSNorm scale that actually multiplies activations.

    FSDP ``lm_head.weight`` is often a CPU FlatParam, so it is not a reliable
    compute-device signal.  Qwen2 RMSNorm mixes ``self.weight`` in ``forward``.
    """
    weight = _rms_norm_weight(final_norm)
    return weight.device if weight is not None else fallback


def _rms_norm_eps(final_norm: nn.Module) -> float:
    inner = _unwrap_module(final_norm)
    for name in ("variance_epsilon", "eps", "epsilon"):
        if hasattr(inner, name):
            return float(getattr(inner, name))
        if hasattr(final_norm, name):
            return float(getattr(final_norm, name))
    return 1e-6


def _use_host_readout(residuals: Tensor) -> bool:
    """Use CPU RMSNorm/unembedding only when residuals are already parked on CPU.

    AllBlocks teacher RM may ``.to("cpu")`` the stacked repr before readout.
    Three-layer jobs keep GPU residuals; do not treat ``requires_grad=False``
    (detached labels) as a reason to move them.
    """
    return residuals.device.type == "cpu"


def _apply_final_norm(final_norm: nn.Module, hidden: Tensor) -> Tensor:
    """RMSNorm on ``hidden``'s device; copy only the tiny scale tensor.

    Do not lift CPU-parked residuals onto CUDA ``self.weight`` (device mix) and
    do not run the teacher unembedding on GPU.
    """
    weight = _rms_norm_weight(final_norm)
    if weight is None:
        return final_norm(hidden)
    if weight.device == hidden.device:
        return final_norm(hidden)
    scale = weight.detach().to(device=hidden.device, dtype=torch.float32)
    x = hidden.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + _rms_norm_eps(final_norm))
    return (scale * x).to(dtype=hidden.dtype)


def _aligned_residual_slice(
    residuals: Tensor,
    final_norm: nn.Module,
    ell: int,
    start: int,
    end: int,
    *,
    host_readout: bool = False,
) -> Tensor:
    slice_ = residuals[:, ell : ell + 1, start:end, :]
    if host_readout:
        if slice_.device.type != "cpu":
            slice_ = slice_.to("cpu")
        return slice_
    device = _rms_norm_device(final_norm, slice_.device)
    if slice_.device != device:
        slice_ = slice_.to(device, non_blocking=device.type == "cuda")
    return slice_


def _host_lm_head_weights(lm_head: nn.Module) -> tuple[Tensor, Tensor | None]:
    """CPU copy of the unembedding for host readout (never a CUDA dtype clone)."""
    inner = _unwrap_module(lm_head)
    weight = inner.weight if hasattr(inner, "weight") else lm_head.weight
    weight_host = weight.detach().to("cpu")
    if weight_host.device.type != "cpu":
        raise RuntimeError(
            f"host lm_head copy stayed on {weight_host.device}; refusing CUDA F.linear"
        )
    bias = getattr(inner, "bias", None)
    if bias is None:
        bias = getattr(lm_head, "bias", None)
    if bias is not None:
        bias = bias.detach().to("cpu")
    return weight_host, bias


def _lm_head_on_hidden(hidden: Tensor, lm_head: nn.Module) -> tuple[Tensor, Tensor, Tensor | None]:
    """Match activation device/dtype to the live unembedding without cloning it."""
    inner = _unwrap_module(lm_head)
    weight = inner.weight if hasattr(inner, "weight") else lm_head.weight
    if hidden.device != weight.device:
        hidden = hidden.to(weight.device)
    if hidden.dtype != weight.dtype:
        hidden = hidden.to(dtype=weight.dtype)
    bias = getattr(inner, "bias", None)
    if bias is None:
        bias = getattr(lm_head, "bias", None)
    if bias is not None and (bias.device != hidden.device or bias.dtype != hidden.dtype):
        bias = bias.to(device=hidden.device, dtype=hidden.dtype)
    return hidden, weight, bias


def _park_residuals_if_host(residuals: Tensor) -> Tensor:
    """No-op unless the caller already parked on CPU (AllBlocks RM)."""
    return residuals


def layerwise_log_probs_on_policy_topk(
    residuals: Tensor,
    policy_topk_ids: Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_chunk_size: int = 32,
    temperature: float = 1.0,
    detach_lm_head: bool = False,
) -> Tensor:
    """Evaluate intermediate readouts on the *final-policy* top-k support.

    ``policy_topk_ids`` is the student final next-token top-k support produced
    by the ordinary OPRD rollout.  We stream the full-vocabulary normalizer in
    token chunks, returning only ``[B, L, T, K]`` log-probabilities.  Thus a
    layerwise auxiliary can cover OPRD's last-2,000-token window without
    materializing a ``[B, L, T, V]`` tensor.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if policy_topk_ids.dim() != 3 or policy_topk_ids.shape[:2] != residuals.shape[:1] + residuals.shape[2:3]:
        raise ValueError(
            "policy_topk_ids must have [B,T,K] matching residuals [B,L,T,D], got "
            f"{tuple(policy_topk_ids.shape)} and {tuple(residuals.shape)}"
        )
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not hasattr(lm_head, "weight"):
        raise ValueError("LayerwiseOPD requires an lm_head weight")

    # Same layer×token stream as per-layer support. If residuals are still on
    # CUDA (7/13/19), RMSNorm stays on GPU. If AllBlocks parked them on CPU,
    # ``_apply_final_norm`` copies only the scale — never ``final_norm(cpu)``
    # against CUDA ``self.weight``.
    from torch.utils.checkpoint import checkpoint

    num_layers = residuals.size(1)
    host_readout = _use_host_readout(residuals)
    ids = policy_topk_ids.to(dtype=torch.long)
    token_chunks = []
    return_device = (
        torch.device("cpu") if host_readout else _rms_norm_device(final_norm, residuals.device)
    )
    host_weight = host_bias = None
    if host_readout:
        host_weight, host_bias = _host_lm_head_weights(lm_head)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def read_one(hidden, weight, head_bias, gather_ids):
        logits = F.linear(hidden, weight, head_bias).float() / temperature
        log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits.gather(-1, gather_ids) - log_norm

    for start in range(0, residuals.size(2), token_chunk_size):
        end = min(start + token_chunk_size, residuals.size(2))
        layer_chunks = []
        gather_ids = ids[:, start:end].unsqueeze(1)
        for ell in range(num_layers):
            hidden = _apply_final_norm(
                final_norm,
                _aligned_residual_slice(
                    residuals, final_norm, ell, start, end, host_readout=host_readout
                ),
            )
            if host_weight is not None:
                if hidden.device.type != "cpu":
                    raise RuntimeError(
                        f"host readout hidden is on {hidden.device}; refusing CUDA F.linear"
                    )
                hidden = hidden.detach().to(dtype=host_weight.dtype)
                w, b = host_weight, host_bias
            else:
                hidden, w, b = _lm_head_on_hidden(hidden, lm_head)
            if detach_lm_head:
                # Preserve the gradient into the intermediate hidden state,
                # but exclude this auxiliary readout from LM-head gradients.
                # The separate final-policy OPD path remains fully trainable.
                w = w.detach()
                b = b.detach() if b is not None else None
            if w.device.type == "cuda" and host_readout:
                raise RuntimeError("host readout still using CUDA lm_head")
            gather = gather_ids.to(device=hidden.device)
            if torch.is_grad_enabled() and not host_readout:
                selected = checkpoint(read_one, hidden, w, b, gather, use_reentrant=False)
            else:
                selected = read_one(hidden, w, b, gather)
            if not torch.is_grad_enabled():
                selected = selected.to("cpu")
            layer_chunks.append(selected)
            del hidden, selected
        token_chunks.append(torch.cat(layer_chunks, dim=1))
    out = torch.cat(token_chunks, dim=2)
    if out.device != return_device:
        out = out.to(device=return_device, dtype=torch.float32)
    return out


def layerwise_topk_log_probs(
    residuals: Tensor,
    *,
    top_k: int,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_chunk_size: int = 32,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return each intermediate readout's own top-k ids and full-vocab log-probs.

    The vocabulary projection is streamed over token chunks, so this never
    materializes a ``[B, L, T, V]`` tensor.  Returned ids/log-probs both have
    shape ``[B, L, T, K]``.  Top-k ids are intentionally non-differentiable;
    callers freeze them at the rollout-policy snapshot before teacher labels
    and PPO updates are computed.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not hasattr(lm_head, "weight"):
        raise ValueError("LayerwiseOPD requires an lm_head weight")

    # Stream over layers and tokens — AllBlocks self-support otherwise builds
    # [B, L, C, V] logits for L≈27 in one shot (same peak that OOMs RM readout).
    num_layers = residuals.size(1)
    residuals = _park_residuals_if_host(residuals)
    host_readout = _use_host_readout(residuals)
    ids_token_chunks, log_prob_token_chunks = [], []
    return_device = (
        torch.device("cpu") if host_readout else _rms_norm_device(final_norm, residuals.device)
    )
    host_weight = host_bias = None
    if host_readout:
        host_weight, host_bias = _host_lm_head_weights(lm_head)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for start in range(0, residuals.size(2), token_chunk_size):
        end = min(start + token_chunk_size, residuals.size(2))
        ids_layers, logp_layers = [], []
        for ell in range(num_layers):
            hidden = _apply_final_norm(
                final_norm,
                _aligned_residual_slice(
                    residuals, final_norm, ell, start, end, host_readout=host_readout
                ),
            )
            if host_weight is not None:
                if hidden.device.type != "cpu":
                    raise RuntimeError(
                        f"host readout hidden is on {hidden.device}; refusing CUDA F.linear"
                    )
                hidden = hidden.detach().to(dtype=host_weight.dtype)
                w, b = host_weight, host_bias
            else:
                hidden, w, b = _lm_head_on_hidden(hidden, lm_head)
            if w.device.type == "cuda" and host_readout:
                raise RuntimeError("host readout still using CUDA lm_head")
            logits = F.linear(hidden, w, b).float() / temperature
            log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
            top_values, top_ids = logits.topk(top_k, dim=-1)
            if not torch.is_grad_enabled():
                ids_layers.append(top_ids.detach().to("cpu"))
                logp_layers.append((top_values - log_norm).to("cpu"))
            else:
                ids_layers.append(top_ids.detach())
                logp_layers.append(top_values - log_norm)
            del hidden, logits, log_norm, top_values, top_ids
        ids_token_chunks.append(torch.cat(ids_layers, dim=1))
        log_prob_token_chunks.append(torch.cat(logp_layers, dim=1))
    ids = torch.cat(ids_token_chunks, dim=2)
    logp = torch.cat(log_prob_token_chunks, dim=2)
    if ids.device != return_device:
        ids = ids.to(device=return_device)
        logp = logp.to(device=return_device, dtype=torch.float32)
    return ids, logp


def layerwise_log_probs_on_per_layer_topk(
    residuals: Tensor,
    layerwise_topk_ids: Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_chunk_size: int = 32,
    temperature: float = 1.0,
) -> Tensor:
    """Evaluate readouts on frozen per-layer student top-k support.

    ``layerwise_topk_ids`` must be ``[B, L, T, K]`` and is normally computed
    from the rollout-policy student before teacher labels are generated.
    Complete-vocabulary normalisation is retained exactly as in
    :func:`layerwise_log_probs_on_policy_topk`.
    """
    if residuals.dim() != 4:
        raise ValueError(f"expected residuals [B,L,T,D], got {tuple(residuals.shape)}")
    expected_prefix = residuals.shape[:3]
    if layerwise_topk_ids.dim() != 4 or tuple(layerwise_topk_ids.shape[:3]) != tuple(expected_prefix):
        raise ValueError(
            "layerwise_topk_ids must be [B,L,T,K] matching residuals [B,L,T,D], got "
            f"{tuple(layerwise_topk_ids.shape)} and {tuple(residuals.shape)}"
        )
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not hasattr(lm_head, "weight"):
        raise ValueError("LayerwiseOPD requires an lm_head weight")

    # Stream over layers *and* token chunks so AllBlocks (L≈27) never materializes
    # a full [B, L, C, V] logits tensor. Detached residuals (teacher RM labels)
    # stay on CPU for RMSNorm + unembedding; do not lift onto CUDA self.weight.
    num_layers = residuals.size(1)
    residuals = _park_residuals_if_host(residuals)
    host_readout = _use_host_readout(residuals)
    ids = layerwise_topk_ids.to(dtype=torch.long)
    token_chunks = []
    return_device = (
        torch.device("cpu") if host_readout else _rms_norm_device(final_norm, residuals.device)
    )
    host_weight = host_bias = None
    if host_readout:
        host_weight, host_bias = _host_lm_head_weights(lm_head)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for start in range(0, residuals.size(2), token_chunk_size):
        end = min(start + token_chunk_size, residuals.size(2))
        layer_chunks = []
        for ell in range(num_layers):
            hidden = _apply_final_norm(
                final_norm,
                _aligned_residual_slice(
                    residuals, final_norm, ell, start, end, host_readout=host_readout
                ),
            )
            if host_weight is not None:
                if hidden.device.type != "cpu":
                    raise RuntimeError(
                        f"host readout hidden is on {hidden.device}; refusing CUDA F.linear"
                    )
                hidden = hidden.detach().to(dtype=host_weight.dtype)
                w, b = host_weight, host_bias
            else:
                hidden, w, b = _lm_head_on_hidden(hidden, lm_head)
            if w.device.type == "cuda" and host_readout:
                raise RuntimeError("host readout still using CUDA lm_head")
            logits = F.linear(hidden, w, b).float() / temperature
            log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
            gather_ids = ids[:, ell : ell + 1, start:end].to(device=hidden.device)
            selected = logits.gather(-1, gather_ids) - log_norm
            if not torch.is_grad_enabled():
                selected = selected.to("cpu")
            layer_chunks.append(selected)
            del hidden, logits, log_norm, gather_ids
        token_chunks.append(torch.cat(layer_chunks, dim=1))
    out = torch.cat(token_chunks, dim=2)
    if out.device != return_device:
        out = out.to(device=return_device, dtype=torch.float32)
    return out


def jlens_log_probs_on_policy_topk(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
    policy_topk_ids: Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_chunk_size: int = 32,
    temperature: float = 1.0,
) -> Tensor:
    """JLens transport then LayerwiseOPD-style top-k log-probs ``[B, L, T, K]``.

    Applies frozen/refreshed ``J`` per layer, then streams Norm+LM-head over the
    *final-policy* top-k support.  Gradients flow through ``residuals`` (and the
    linear readout) when they require grad; ``J`` stays a frozen buffer.
    """
    transported = jlens_transported_residuals(residuals, artifacts=artifacts)
    return layerwise_log_probs_on_policy_topk(
        transported,
        policy_topk_ids,
        final_norm=final_norm,
        lm_head=lm_head,
        token_chunk_size=token_chunk_size,
        temperature=temperature,
    )


def jlens_log_probs_on_per_layer_topk(
    residuals: Tensor,
    *,
    artifacts: JLensArtifacts,
    layerwise_topk_ids: Tensor,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_chunk_size: int = 32,
    temperature: float = 1.0,
) -> Tensor:
    """JLens transport then evaluate frozen per-layer student top-k support."""
    transported = jlens_transported_residuals(residuals, artifacts=artifacts)
    return layerwise_log_probs_on_per_layer_topk(
        transported,
        layerwise_topk_ids=layerwise_topk_ids,
        final_norm=final_norm,
        lm_head=lm_head,
        token_chunk_size=token_chunk_size,
        temperature=temperature,
    )


def _accumulate_masked_jalign_kl_chunk(
    teacher_logits_chunk: Tensor,
    student_logits_chunk: Tensor,
    weights_chunk: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(weighted_kl_sum, weighted_teacher_entropy_sum)`` for one token chunk."""
    teacher_log_probs = F.log_softmax(teacher_logits_chunk.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits_chunk.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    return token_kl.mul(weights_chunk).sum(), entropy.mul(weights_chunk).sum()


def masked_jalign_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    position_mask: Tensor,
    *,
    temperature: float,
    token_chunk_size: int = 32,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return ``T^2 KL(teacher || student)`` over selected response positions.

    Softmax / KL are streamed over the token axis in chunks of ``token_chunk_size``
    so a ``last_k=2000`` full-vocabulary objective does not materialize a second
    full ``[B,L,T,V]`` float32 probability tensor.  The scalar loss is identical
    to the unchunked reduction.
    """
    if temperature <= 0:
        raise ValueError("JAlign temperature must be positive")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if teacher_logits.shape != student_logits.shape or teacher_logits.dim() != 4:
        raise ValueError(
            "teacher/student J-logits must have the same [B,L,T,V] shape, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if position_mask.shape != teacher_logits.shape[:1] + teacher_logits.shape[2:3]:
        raise ValueError(
            f"position mask {tuple(position_mask.shape)} does not match J-logits {tuple(teacher_logits.shape)}"
        )
    num_layers = teacher_logits.size(1)
    weights = position_mask.to(dtype=torch.float32, device=student_logits.device).unsqueeze(1)
    denominator = weights.sum() * num_layers
    if denominator.item() == 0:
        raise ValueError("JAlign position mask selects no valid response tokens")

    kl_sum = student_logits.new_zeros(())
    entropy_sum = student_logits.new_zeros(())
    seq_len = teacher_logits.size(2)
    for start in range(0, seq_len, token_chunk_size):
        end = min(start + token_chunk_size, seq_len)
        teacher_chunk = teacher_logits[:, :, start:end]
        if teacher_chunk.device != student_logits.device:
            teacher_chunk = teacher_chunk.to(device=student_logits.device, non_blocking=True)
        chunk_kl, chunk_entropy = _accumulate_masked_jalign_kl_chunk(
            teacher_chunk,
            student_logits[:, :, start:end],
            weights[:, :, start:end],
            temperature=temperature,
        )
        kl_sum = kl_sum + chunk_kl
        entropy_sum = entropy_sum + chunk_entropy

    loss = kl_sum / denominator * (temperature**2)
    metrics = {
        "jalign/valid_tokens": position_mask.sum().detach(),
        "jalign/mean_kl": loss.detach(),
        "jalign/teacher_entropy": (entropy_sum / denominator).detach(),
    }
    return loss, metrics


def masked_jalign_kl_from_transported_residuals(
    teacher_logits: Tensor,
    student_residuals: Tensor,
    position_mask: Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    token_ids: Tensor | None = None,
    temperature: float,
    token_chunk_size: int = 32,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Same JAlign KL as ``masked_jalign_kl``, streaming the student LM-head.

    ``student_residuals`` are already Jacobian-transported ``[B,L,T,D]``.  Teacher
    logits may stay on CPU; each token chunk is moved to the student device only
    for the softmax/KL.  This avoids materializing a full student ``[B,L,T,V]``.
    """
    if temperature <= 0:
        raise ValueError("JAlign temperature must be positive")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if student_residuals.dim() != 4:
        raise ValueError(f"expected student residuals [B,L,T,D], got {tuple(student_residuals.shape)}")
    if teacher_logits.dim() != 4:
        raise ValueError(f"expected teacher logits [B,L,T,V], got {tuple(teacher_logits.shape)}")
    if teacher_logits.shape[:3] != student_residuals.shape[:3]:
        raise ValueError(
            "teacher logits and student residuals must share [B,L,T], got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_residuals.shape)}"
        )
    if position_mask.shape != student_residuals.shape[:1] + student_residuals.shape[2:3]:
        raise ValueError(
            f"position mask {tuple(position_mask.shape)} does not match residuals {tuple(student_residuals.shape)}"
        )

    num_layers = student_residuals.size(1)
    weights = position_mask.to(dtype=torch.float32, device=student_residuals.device).unsqueeze(1)
    denominator = weights.sum() * num_layers
    if denominator.item() == 0:
        raise ValueError("JAlign position mask selects no valid response tokens")

    kl_sum = student_residuals.new_zeros(())
    entropy_sum = student_residuals.new_zeros(())
    seq_len = student_residuals.size(2)
    for start in range(0, seq_len, token_chunk_size):
        end = min(start + token_chunk_size, seq_len)
        student_chunk = readout_selected_logits_from_residuals(
            student_residuals[:, :, start:end],
            final_norm=final_norm,
            lm_head=lm_head,
            token_ids=token_ids,
        )
        teacher_chunk = teacher_logits[:, :, start:end]
        if teacher_chunk.device != student_chunk.device:
            teacher_chunk = teacher_chunk.to(device=student_chunk.device, non_blocking=True)
        chunk_kl, chunk_entropy = _accumulate_masked_jalign_kl_chunk(
            teacher_chunk,
            student_chunk,
            weights[:, :, start:end],
            temperature=temperature,
        )
        kl_sum = kl_sum + chunk_kl
        entropy_sum = entropy_sum + chunk_entropy

    loss = kl_sum / denominator * (temperature**2)
    metrics = {
        "jalign/valid_tokens": position_mask.sum().detach(),
        "jalign/mean_kl": loss.detach(),
        "jalign/teacher_entropy": (entropy_sum / denominator).detach(),
    }
    return loss, metrics


def masked_jalign_mse(
    teacher_vectors: Tensor,
    student_vectors: Tensor,
    position_mask: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """L2-normalized MSE on transported J-vectors ``[B, L, T, D]``.

    Matches OPRD's ``normalized_mse_loss`` aggregation: per-layer token-mean
    MSE after unit-sphere normalization, then mean over layers.  Teacher
    vectors are detached.
    """
    if teacher_vectors.shape != student_vectors.shape or teacher_vectors.dim() != 4:
        raise ValueError(
            "teacher/student J-vectors must have the same [B,L,T,D] shape, got "
            f"{tuple(teacher_vectors.shape)} and {tuple(student_vectors.shape)}"
        )
    if position_mask.shape != teacher_vectors.shape[:1] + teacher_vectors.shape[2:3]:
        raise ValueError(
            f"position mask {tuple(position_mask.shape)} does not match J-vectors {tuple(teacher_vectors.shape)}"
        )
    if int(position_mask.sum().item()) == 0:
        raise ValueError("JAlign-MSE position mask selects no valid response tokens")

    student_norm = F.normalize(student_vectors.float(), p=2, dim=-1)
    teacher_norm = F.normalize(teacher_vectors.detach().float(), p=2, dim=-1)
    per_token = ((student_norm - teacher_norm) ** 2).mean(dim=-1)  # [B, L, T]
    weights = position_mask.to(dtype=per_token.dtype).unsqueeze(1)
    denominator = weights.sum() * per_token.size(1)
    loss = per_token.mul(weights).sum() / denominator
    # Cosine on the same unit vectors (diagnostic only).
    per_token_cos = (student_norm * teacher_norm).sum(dim=-1)
    mean_cos = per_token_cos.mul(weights).sum() / denominator
    metrics = {
        "jalign/valid_tokens": position_mask.sum().detach(),
        "jalign/mean_mse": loss.detach(),
        "jalign/mean_cosine": mean_cos.detach(),
    }
    return loss, metrics


def masked_direct_softmax_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    position_mask: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Same KL as JAlign, with metrics explicitly labelled for its no-JLens control."""
    loss, metrics = masked_jalign_kl(teacher_logits, student_logits, position_mask, temperature=temperature)
    return loss, {key.replace("jalign/", "direct_softmax/"): value for key, value in metrics.items()}


def masked_layerwise_opd_policy_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    position_mask: Tensor,
    *,
    top_k: int,
    temperature: float,
) -> Tensor:
    """Student-top-k, score-function-style OPD objective on intermediate readouts.

    The teacher/student logits are normalized over their complete shared
    vocabulary, but the detached pseudo-advantage is constructed only on the
    student's top-k support.  This mirrors the existing ``only_stu`` OPD
    update while treating each selected intermediate readout as a pseudo-policy.
    """
    if temperature <= 0:
        raise ValueError("Layerwise OPD temperature must be positive")
    if top_k <= 0 or top_k > student_logits.size(-1):
        raise ValueError(f"Layerwise OPD top_k must be in [1, {student_logits.size(-1)}], got {top_k}")
    if teacher_logits.shape != student_logits.shape or teacher_logits.dim() != 4:
        raise ValueError("teacher/student logits must share [B,L,T,V] shape")
    if position_mask.shape != student_logits.shape[:1] + student_logits.shape[2:3]:
        raise ValueError("position mask must have shape [B,T]")

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    candidate_ids = student_logits.topk(top_k, dim=-1).indices
    student_candidate_log_probs = student_log_probs.gather(-1, candidate_ids)
    teacher_candidate_log_probs = teacher_log_probs.gather(-1, candidate_ids)

    # This is the detached only_stu OPD pseudo-reward: -w_S(log p_S-log p_T).
    # The loss below is -A * log p_S, so gradients enter only through the final
    # score term, rather than differentiating through the pseudo-advantage.
    weights = F.softmax(student_candidate_log_probs.detach(), dim=-1)
    score_coeff = (weights * (student_candidate_log_probs.detach() - teacher_candidate_log_probs.detach())).detach()
    per_layer_token_loss = (score_coeff * student_candidate_log_probs).sum(dim=-1)
    mask = position_mask.to(per_layer_token_loss.dtype).unsqueeze(1)
    denominator = mask.sum() * per_layer_token_loss.size(1)
    if denominator.item() == 0:
        raise ValueError("Layerwise OPD position mask selects no valid response tokens")
    return (per_layer_token_loss * mask).sum() / denominator


def masked_layerwise_opd_policy_losses_from_log_probs(
    teacher_candidate_log_probs: Tensor,
    student_candidate_log_probs: Tensor,
    position_mask: Tensor,
    domain_is_code: Tensor | None = None,
) -> Tensor:
    """Per-layer OPD score-function auxiliaries on final-policy top-k support."""
    if teacher_candidate_log_probs.shape != student_candidate_log_probs.shape:
        raise ValueError("teacher/student layerwise candidate log-probs must share a shape")
    if student_candidate_log_probs.dim() != 4:
        raise ValueError("layerwise candidate log-probs must have [B,L,T,K] shape")
    if position_mask.shape != student_candidate_log_probs.shape[:1] + student_candidate_log_probs.shape[2:3]:
        raise ValueError("position mask must have [B,T] shape")

    weights = F.softmax(student_candidate_log_probs.detach(), dim=-1)
    score_coeff = (
        weights * (student_candidate_log_probs.detach() - teacher_candidate_log_probs.detach())
    ).detach()
    per_layer_token_loss = (score_coeff * student_candidate_log_probs).sum(dim=-1)
    mask = position_mask.to(per_layer_token_loss.dtype).unsqueeze(1)
    if domain_is_code is not None:
        from verl.trainer.ppo.core_algos import agg_loss_domain_equal_token_mean

        layer_losses = []
        token_mask = position_mask.to(per_layer_token_loss.dtype)
        for layer_idx in range(per_layer_token_loss.size(1)):
            layer_losses.append(
                agg_loss_domain_equal_token_mean(
                    per_layer_token_loss[:, layer_idx, :],
                    token_mask,
                    domain_is_code,
                )
            )
        return torch.stack(layer_losses)

    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("LayerwiseOPD position mask selects no valid response tokens")
    return (per_layer_token_loss * mask).sum(dim=(0, 2)) / denominator


def masked_layerwise_transition_opd_policy_losses_from_log_probs(
    teacher_layer_log_probs: Tensor,
    student_layer_log_probs: Tensor,
    teacher_final_log_probs: Tensor,
    student_final_log_probs: Tensor,
    position_mask: Tensor,
    *,
    delta_align: bool = False,
    reverse_transition_kl: bool = False,
    diagnostics: dict | None = None,
    weight_only: bool = False,
    center_only: bool = False,
) -> Tensor:
    """Score-function OPD losses on depth transitions.

    The intermediate and final readouts must use the same candidate support.
    Each returned loss supervises one transition ``7->13``, ``13->19``, ...
    rather than matching an absolute layer distribution.
    """
    if teacher_layer_log_probs.shape != student_layer_log_probs.shape:
        raise ValueError("teacher/student transition layer log-probs must share a shape")
    if teacher_layer_log_probs.dim() != 4:
        raise ValueError("transition layer log-probs must have [B,L,T,K] shape")
    expected_final = teacher_layer_log_probs.shape[:1] + teacher_layer_log_probs.shape[2:]
    if teacher_final_log_probs.shape != expected_final or student_final_log_probs.shape != expected_final:
        raise ValueError("final transition log-probs must have [B,T,K] shape matching layer terms")
    if position_mask.shape != teacher_layer_log_probs.shape[:1] + teacher_layer_log_probs.shape[2:3]:
        raise ValueError("position mask must have [B,T] shape")

    teacher_path = torch.cat([teacher_layer_log_probs, teacher_final_log_probs.unsqueeze(1)], dim=1)
    student_path = torch.cat([student_layer_log_probs, student_final_log_probs.unsqueeze(1)], dim=1)
    teacher_delta = (teacher_path[:, 1:] - teacher_path[:, :-1]).detach()
    student_delta = student_path[:, 1:] - student_path[:, :-1]

    if weight_only and center_only:
        raise ValueError("weight_only and center_only are mutually exclusive")
    if weight_only:
        weights = F.softmax(student_delta.detach(), dim=-1)
        score_coeff = (weights * (student_delta.detach() - teacher_delta)).detach()
        if diagnostics is not None:
            diagnostics["actor/weight_only_enabled"] = 1.0
    elif center_only:
        weights = F.softmax(student_path[:, 1:].detach(), dim=-1)
        q_s = F.softmax(student_delta.detach(), dim=-1)
        reward = teacher_delta - student_delta.detach()
        baseline = (q_s * reward).sum(dim=-1, keepdim=True)
        score_coeff = -(weights * (reward - baseline)).detach()
        if diagnostics is not None:
            diagnostics["actor/center_only_enabled"] = 1.0
    elif reverse_transition_kl:
        if delta_align:
            raise ValueError("reverse transition KL cannot be combined with delta alignment")
        # Both weights and centered rewards are frozen score coefficients.
        # dL/d(delta_S) = -q_S * (R - E_qS[R]) = d KL(q_S || q_T)/d(delta_S).
        weights = F.softmax(student_delta.detach(), dim=-1)
        teacher_weights = F.softmax(teacher_delta, dim=-1)
        reward = teacher_delta - student_delta.detach()
        baseline = (weights * reward).sum(dim=-1, keepdim=True)
        centered_reward = reward - baseline
        score_coeff = -(weights * centered_reward).detach()
        for name, value in (("q_s", weights), ("q_t", teacher_weights),
                            ("baseline", baseline), ("centered_reward", centered_reward)):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite reverse transition KL {name}")
            if diagnostics is not None:
                diagnostics[f"actor/reverse_transition_kl_{name}_finite"] = 1.0
        if diagnostics is not None:
            diagnostics["actor/reverse_transition_kl_enabled"] = 1.0
    else:
        # Preserve the historical destination-weighted surrogate exactly.
        weights = F.softmax(student_path[:, 1:].detach(), dim=-1)
        score_coeff = (weights * (student_delta.detach() - teacher_delta)).detach()
    per_transition_token_loss = (score_coeff * student_delta).sum(dim=-1)
    if delta_align:
        from verl.utils.depth_delta_align import transition_alignment_weights
        routing = transition_alignment_weights(teacher_delta, student_delta)
        # Caller averages L returned terms: preserve the original loss scale.
        per_transition_token_loss = per_transition_token_loss * routing * routing.shape[1]
    mask = position_mask.to(per_transition_token_loss.dtype).unsqueeze(1)
    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("transition OPD position mask selects no valid response tokens")
    return (per_transition_token_loss * mask).sum(dim=(0, 2)) / denominator


def final_teacher_log_probs_as_layerwise_targets(
    teacher_final_candidate_log_probs: Tensor,
    response_mask: Tensor,
    *,
    positions: str,
    last_k: int,
    first_k: int,
    num_layers: int,
) -> Tensor:
    """Broadcast final-teacher OPD labels to intermediate student readouts.

    Inputs are the ordinary final-teacher log probabilities evaluated on the
    frozen final-student support.  This changes only the teacher target used by
    LayerwiseOPD; support, student readout, and final OPD remain unchanged.
    """
    if teacher_final_candidate_log_probs.dim() != 3:
        raise ValueError("final teacher candidate log-probs must have [B,T,K] shape")
    if response_mask.shape != teacher_final_candidate_log_probs.shape[:2]:
        raise ValueError("response_mask must match final teacher [B,T,K]")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    from verl.utils.rep_distillation import compact_response_repr_by_positions

    compact, _ = compact_response_repr_by_positions(
        teacher_final_candidate_log_probs,
        response_mask,
        positions,
        last_k,
        first_k,
    )
    return compact.unsqueeze(1).expand(-1, num_layers, -1, -1)


def masked_layerwise_opd_policy_loss_from_log_probs(
    teacher_candidate_log_probs: Tensor,
    student_candidate_log_probs: Tensor,
    position_mask: Tensor,
) -> Tensor:
    """Mean OPD score-function auxiliary on final-policy top-k support."""
    return masked_layerwise_opd_policy_losses_from_log_probs(
        teacher_candidate_log_probs,
        student_candidate_log_probs,
        position_mask,
    ).mean()


def layerwise_opd_candidate_distribution_diagnostics(
    teacher_candidate_log_probs: Tensor,
    student_candidate_log_probs: Tensor,
    position_mask: Tensor,
) -> dict[str, Tensor]:
    """Return per-layer diagnostics on the shared candidate support.

    These values are detached observational statistics and are never part of
    the training objective. ``candidate_kl_contrib`` is the teacher-weighted
    contribution on the retained candidates, not a full-vocabulary KL; it may
    therefore be negative when omitted vocabulary mass differs.
    """
    if teacher_candidate_log_probs.shape != student_candidate_log_probs.shape:
        raise ValueError("teacher/student layerwise candidate log-probs must share a shape")
    if student_candidate_log_probs.dim() != 4:
        raise ValueError("layerwise candidate log-probs must have [B,L,T,K] shape")
    if position_mask.shape != student_candidate_log_probs.shape[:1] + student_candidate_log_probs.shape[2:3]:
        raise ValueError("position mask must have [B,T] shape")

    mask = position_mask.to(student_candidate_log_probs.dtype).unsqueeze(1).unsqueeze(-1)
    token_denominator = mask.sum()
    if token_denominator.item() == 0:
        raise ValueError("LayerwiseOPD position mask selects no valid response tokens")

    teacher_probs = teacher_candidate_log_probs.detach().exp()
    student_probs = student_candidate_log_probs.detach().exp()
    normalized_student = torch.softmax(student_candidate_log_probs.detach(), dim=-1)
    candidate_entropy = -(normalized_student * normalized_student.clamp_min(1e-12).log()).sum(dim=-1)
    token_mask = mask.squeeze(-1)
    return {
        "teacher_candidate_mass": (teacher_probs * mask).sum(dim=(0, 2, 3)) / token_denominator,
        "student_candidate_mass": (student_probs * mask).sum(dim=(0, 2, 3)) / token_denominator,
        "candidate_kl_contrib": (
            teacher_probs * (teacher_candidate_log_probs.detach() - student_candidate_log_probs.detach()) * mask
        ).sum(dim=(0, 2, 3))
        / token_denominator,
        "mean_abs_logprob_gap": (
            (teacher_candidate_log_probs.detach() - student_candidate_log_probs.detach()).abs() * mask
        ).sum(dim=(0, 2, 3))
        / (token_denominator * student_candidate_log_probs.size(-1)),
        "student_candidate_entropy": (candidate_entropy * token_mask).sum(dim=(0, 2)) / token_denominator,
    }


def should_run_layerwise_opd_diagnostics(*, enabled: bool, interval: int, global_step: int) -> bool:
    """Gate diagnostic-only work on the authoritative global training step."""
    if interval <= 0:
        raise ValueError("layerwise_opd_diagnostics_interval must be positive")
    if not enabled:
        return False
    if global_step < 0:
        raise ValueError("layerwise_opd_diagnostics requires meta_info['global_steps']")
    return global_step % interval == 0


def masked_jflow_opd_policy_losses_from_log_probs(
    teacher_current_log_probs: Tensor,
    teacher_anchor_log_probs: Tensor,
    student_current_log_probs: Tensor,
    student_anchor_log_probs: Tensor,
    position_mask: Tensor,
    *,
    token_offset: int,
) -> Tensor:
    """Per-layer OPD auxiliaries on temporal changes in J-space log-probs.

    The previous position is a detached anchor, so credit enters only through
    the current token's J-space score term.
    """
    shapes = {
        tuple(teacher_current_log_probs.shape),
        tuple(teacher_anchor_log_probs.shape),
        tuple(student_current_log_probs.shape),
        tuple(student_anchor_log_probs.shape),
    }
    if len(shapes) != 1:
        raise ValueError("all teacher/student J-Flow log-probs must share a shape")
    if student_current_log_probs.dim() != 4:
        raise ValueError("J-Flow candidate log-probs must have [B,L,T,K] shape")
    expected_full_width = student_current_log_probs.size(2) + token_offset
    if position_mask.shape != (student_current_log_probs.size(0), expected_full_width):
        raise ValueError("J-Flow position mask must cover current and anchor positions")
    if token_offset <= 0:
        raise ValueError(
            f"J-Flow token_offset must be positive, got {token_offset}"
        )

    teacher_delta = (teacher_current_log_probs - teacher_anchor_log_probs).detach()
    student_delta = student_current_log_probs - student_anchor_log_probs.detach()
    weights = F.softmax(student_current_log_probs.detach(), dim=-1)
    score_coeff = (weights * (student_delta.detach() - teacher_delta)).detach()
    per_layer_token_loss = (score_coeff * student_delta).sum(dim=-1)
    flow_mask = (
        position_mask[:, token_offset:].bool()
        & position_mask[:, :-token_offset].bool()
    ).to(per_layer_token_loss.dtype).unsqueeze(1)
    denominator = flow_mask.sum()
    if denominator.item() == 0:
        raise ValueError("J-Flow position mask selects no valid token pairs")
    return (per_layer_token_loss * flow_mask).sum(dim=(0, 2)) / denominator


def aggregate_final_and_layerwise_opd_loss(
    final_policy_loss: Tensor,
    layerwise_loss: Tensor,
    *,
    num_intermediate_layers: int,
    intermediate_layer_coef: float = 1.0,
    aggregation: str = "equal_weight",
) -> Tensor:
    """Combine a final OPD term and the mean intermediate OPD/JOPD loss.

    ``layerwise_loss`` is already the mean of one OPD loss for each selected
    intermediate layer.  Consequently, with ``intermediate_layer_coef=1`` the
    returned objective is

    ``(L_final + sum_l L_l) / (1 + num_intermediate_layers)``.

    With ``aggregation="equal_weight"``, the optional coefficient scales every
    intermediate layer *individually*.  ``aggregation="legacy_additive"``
    restores the historical LayerwiseOPD+OPD objective
    ``L_final + coef * mean_l(L_l)``.
    """
    if num_intermediate_layers <= 0:
        raise ValueError("num_intermediate_layers must be positive")
    if intermediate_layer_coef < 0:
        raise ValueError("intermediate_layer_coef must be non-negative")
    if final_policy_loss.shape != layerwise_loss.shape:
        raise ValueError("final_policy_loss and layerwise_loss must have matching shapes")
    aggregation = str(aggregation).strip().lower()
    if aggregation == "legacy_additive":
        return final_policy_loss + intermediate_layer_coef * layerwise_loss
    if aggregation != "equal_weight":
        raise ValueError(
            f"unknown layerwise OPD aggregation={aggregation!r}; "
            "expected 'equal_weight' or 'legacy_additive'"
        )

    total_intermediate_weight = intermediate_layer_coef * num_intermediate_layers
    return (final_policy_loss + layerwise_loss * total_intermediate_weight) / (1.0 + total_intermediate_weight)
