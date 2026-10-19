# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, domain_is_code_from_sources, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.att_distillation import (
    att_distillation_query_valid_fraction,
    attention_distillation_loss,
    disable_gradient_checkpointing,
    forward_response_attn_rows,
    get_att_distillation_context_metadata,
    validate_att_distillation_loss_type,
)
from verl.utils.rep_distillation import (
    LowRankCrossArchProjector,
    DirectLowRankCrossArchProjector,
    ResidualLowRankCrossArchProjector,
    infer_subspace_mode_from_checkpoint,
    load_preexp_checkpoint_dict,
    resolve_student_projector_settings,
    _squeeze_hidden_rmpad,
    build_compact_rep_distillation_position_mask,
    build_rep_distillation_position_mask,
    compact_response_repr_by_positions,
    extract_teacher_response_hidden_repr,
    forward_response_hidden_repr,
    hidden_states_tuple_to_response_repr,
    compute_rep_alignment_metrics,
    multi_layer_normalized_cosine_similarity,
    multi_layer_normalized_mse_loss,
    select_stacked_block_hidden_states,
    validate_rep_distillation_layers,
    validate_rep_distillation_positions,
    validate_rep_projector_mode,
)
from verl.utils.jlens_distillation import (
    FrozenAffineReadoutArtifacts,
    JLensArtifacts,
    ConditionalResidualJLens,
    aggregate_final_and_layerwise_opd_loss,
    frozen_affine_readout_residuals,
    activation_only_online_jcr_layerwise_loss,
    jacobian_backward_identity_residuals,
    online_jacobian_credit_residuals,
    direct_selected_logits,
    jlens_log_probs_on_policy_topk,
    jlens_log_probs_on_per_layer_topk,
    jlens_identity_credit_residuals,
    jlens_transported_residuals,
    layerwise_log_probs_on_policy_topk,
    layerwise_log_probs_on_per_layer_topk,
    layerwise_topk_log_probs,
    load_selected_vocab_ids,
    masked_direct_softmax_kl,
    masked_jalign_kl_from_transported_residuals,
    masked_jalign_mse,
    masked_jflow_opd_policy_losses_from_log_probs,
    masked_layerwise_opd_policy_loss,
    masked_layerwise_opd_policy_loss_from_log_probs,
    masked_layerwise_opd_policy_losses_from_log_probs,
    masked_layerwise_transition_opd_policy_losses_from_log_probs,
    layerwise_opd_candidate_distribution_diagnostics,
    should_run_layerwise_opd_diagnostics,
    parse_jalign_layers,
    resolve_final_norm_and_lm_head,
    resolve_decoder_blocks,
    resolve_layerwise_opd_gradient_routing,
    sampled_online_jacobian_credit_residuals,
    resolve_jalign_layers,
    stochastic_jlens_transported_residuals,
    validate_jalign_loss_type,
    validate_jalign_gradient_routing,
    validate_jalign_stoch_probe,
    jalign_uses_fitted_lens,
    validate_jalign_student_transport,
    validate_layerwise_opd_sample_one_layer,
)
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    prepare_dynamic_batch,
    restore_dynamic_batch,
)
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig


def _fixed_transition_weights(spec: str, count: int) -> tuple[float, ...] | None:
    """Parse optional, normalized weights for the depth-transition terms."""
    if not spec:
        return None
    weights = tuple(float(part.strip()) for part in spec.split(","))
    if len(weights) != count or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError(f"expected {count} finite nonnegative transition weights, got {spec!r}")
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"transition weights must sum to 1, got {spec!r}")
    return weights


def _clone_saved_fsdp_parameter_view(tensor: torch.Tensor) -> torch.Tensor:
    """Keep parameter data needed by an activation VJP after FSDP resharding."""
    base = tensor
    while getattr(base, "_base", None) is not None:
        base = base._base
    return tensor.clone() if isinstance(base, nn.Parameter) else tensor


def _compute_online_jcr_cotangents(
    actor_module: nn.Module,
    model_inputs: dict,
    response_mask: torch.Tensor,
    *,
    sampled_layer_id: int,
    teacher_layer_slot: int,
    positions: str,
    last_k: int,
    first_k: int,
    temperature: float,
    use_remove_padding: bool,
    use_ulysses_sp: bool,
    ulysses_sequence_parallel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute detached raw and suffix-routed cotangents in a temporary graph."""
    prepass_topk_ids, _ = compact_response_repr_by_positions(
        model_inputs["student_top_k_ids"],
        response_mask,
        positions,
        last_k,
        first_k,
    )
    prepass_mask = build_compact_rep_distillation_position_mask(
        response_mask,
        positions,
        last_k,
        first_k,
    )
    decoder_blocks = resolve_decoder_blocks(actor_module)
    prepass_source_captures = []
    prepass_final_captures = []

    def capture_prepass_source(_module, _inputs, output):
        # The norm prepass only needs the Jacobian of the suffix beginning at
        # the sampled block output.  Detach that output so autograd does not
        # retain the (mathematically irrelevant) prefix graph for this second
        # forward.  Returning the replacement keeps the suffix VJP exact while
        # substantially reducing the prepass peak memory.
        source = output[0] if isinstance(output, tuple) else output
        source = source.detach().requires_grad_(True)
        prepass_source_captures.append(source)
        if isinstance(output, tuple):
            return (source, *output[1:])
        return source

    def capture_prepass_final(_module, inputs):
        prepass_final_captures.append(inputs[0])

    student_final_norm, _ = resolve_final_norm_and_lm_head(actor_module)
    source_hook = decoder_blocks[sampled_layer_id].register_forward_hook(capture_prepass_source)
    final_hook = student_final_norm.register_forward_pre_hook(capture_prepass_final)
    try:
        with torch.autograd.graph.saved_tensors_hooks(
            _clone_saved_fsdp_parameter_view,
            lambda tensor: tensor,
        ):
            forward_response_hidden_repr(
                actor_module,
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
                model_inputs["position_ids"],
                response_mask,
                use_remove_padding=use_remove_padding,
                use_ulysses_sp=use_ulysses_sp,
                ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
                positions=positions,
                layers="last",
                last_k=last_k,
                first_k=first_k,
            )
    finally:
        source_hook.remove()
        final_hook.remove()
    if len(prepass_source_captures) != 1 or len(prepass_final_captures) != 1:
        raise RuntimeError(
            "activation-only Online-JCR prepass expected one source and one "
            f"final capture, got {len(prepass_source_captures)} and "
            f"{len(prepass_final_captures)}"
        )

    source_probe = prepass_source_captures[0].detach().requires_grad_(True)
    compact_probe = extract_teacher_response_hidden_repr(
        source_probe,
        response_mask,
        positions,
        layers="last",
        last_k=last_k,
        first_k=first_k,
        compact=True,
    )
    fsdp_context = (
        FSDP.summon_full_params(actor_module, recurse=False, writeback=False)
        if isinstance(actor_module, FSDP)
        else nullcontext()
    )
    with fsdp_context:
        student_final_norm, student_lm_head = resolve_final_norm_and_lm_head(actor_module)
        probe_log_probs = layerwise_log_probs_on_policy_topk(
            compact_probe.unsqueeze(1),
            policy_topk_ids=prepass_topk_ids,
            final_norm=student_final_norm,
            lm_head=student_lm_head,
            temperature=temperature,
        )
        probe_loss = masked_layerwise_opd_policy_loss_from_log_probs(
            model_inputs["teacher_layerwise_candidate_log_probs"][:,
                teacher_layer_slot : teacher_layer_slot + 1
            ],
            probe_log_probs,
            prepass_mask,
        )
        local_cotangent = torch.autograd.grad(probe_loss, source_probe, create_graph=False)[0]
    routed_cotangent = torch.autograd.grad(
        prepass_final_captures[0],
        prepass_source_captures[0],
        grad_outputs=local_cotangent.to(prepass_final_captures[0].dtype),
        create_graph=False,
    )[0]
    return local_cotangent.detach(), routed_cotangent.detach()


def _compute_activation_only_online_jcr_cotangent(*args, **kwargs) -> torch.Tensor:
    """Compute one detached suffix VJP and release its temporary graph."""
    _, routed_cotangent = _compute_online_jcr_cotangents(*args, **kwargs)
    return routed_cotangent


def _online_jcr_norm_chunk_indices(
    attention_mask: torch.Tensor,
    max_tokens: int,
) -> list[list[int]]:
    """Partition a local micro-batch for the norm-only VJP prepass."""
    batch_size = int(attention_mask.shape[0])
    if batch_size == 0:
        return []
    if max_tokens <= 0:
        return [list(range(batch_size))]

    effective_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    local_chunks = min(
        batch_size,
        max(1, (int(effective_lengths.sum().item()) + max_tokens - 1) // max_tokens),
    )
    synchronized_chunk_count = local_chunks
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        synchronized_chunks = torch.tensor([local_chunks], device=attention_mask.device, dtype=torch.long)
        torch.distributed.all_reduce(synchronized_chunks, op=torch.distributed.ReduceOp.MAX)
        synchronized_chunk_count = int(synchronized_chunks.item())

    workloads = calculate_workload(effective_lengths)
    partitions = get_seqlen_balanced_partitions(workloads, local_chunks, equal_size=False)
    # FSDP ranks must execute the same number of temporary forwards/VJPs. A rank
    # with fewer local samples pads the call sequence; the empty partitions are
    # executed on a throwaway sample and excluded from its norm statistics.
    partitions.extend([[] for _ in range(synchronized_chunk_count - local_chunks)])
    return partitions


def _slice_online_jcr_model_inputs(
    model_inputs: dict,
    indices: list[int],
    batch_size: int,
) -> dict:
    """Select batch-leading tensors while leaving scalar/shared inputs intact."""
    sliced = {}
    index = None
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            if index is None or index.device != value.device:
                index = torch.tensor(indices, device=value.device, dtype=torch.long)
            sliced[key] = value.index_select(0, index)
        else:
            sliced[key] = value
    return sliced


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.layerwise_adaptive_ema = None
        self.layerwise_adaptive_prev_ema = None
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.rep_projector = None
        self._rep_projector_in_optimizer = False
        self.low_rank_projector: LowRankCrossArchProjector | DirectLowRankCrossArchProjector | None = None
        self._low_rank_projector_in_optimizer = False
        self.residual_low_rank_projector: ResidualLowRankCrossArchProjector | None = None
        self._residual_low_rank_projector_in_optimizer = False
        self.conditional_jlens = None
        self._conditional_jlens_in_optimizer = False
        self.identity_adapter_jlens = None
        self._identity_adapter_jlens_in_optimizer = False
        self._layerwise_opd_jacobian_artifacts = None
        self._layerwise_opd_affine_artifacts = None
        self._online_jcr_layer_cursor = 0

    def _get_or_create_low_rank_jlens(
        self, artifacts: JLensArtifacts, *, identity_adapter: bool = False
    ) -> ConditionalResidualJLens:
        module_attr = "identity_adapter_jlens" if identity_adapter else "conditional_jlens"
        optimizer_attr = "_identity_adapter_jlens_in_optimizer" if identity_adapter else "_conditional_jlens_in_optimizer"
        module = getattr(self, module_attr)
        if module is None:
            rank_key = "jalign_identity_adapter_rank" if identity_adapter else "jalign_conditional_rank"
            lr_key = "jalign_identity_adapter_lr" if identity_adapter else "jalign_conditional_lr"
            module = ConditionalResidualJLens(artifacts, rank=int(self.config.get(rank_key, 1))).to(get_device_id())
            setattr(self, module_attr, module)
            # This small module lives outside FSDP; explicitly average its gradients.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                def all_reduce_average(grad: torch.Tensor) -> torch.Tensor:
                    # Rank>1 factors can receive a non-contiguous einsum gradient;
                    # NCCL all_reduce requires a contiguous tensor.
                    grad = grad.contiguous()
                    torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                    return grad

                for parameter in module.parameters():
                    parameter.register_hook(all_reduce_average)
            if self.actor_optimizer is not None:
                self.actor_optimizer.add_param_group({
                    "params": module.parameters(),
                    "lr": float(self.config.get(lr_key, 2e-3)),
                })
                setattr(self, optimizer_attr, True)
        return module

    def _get_or_create_rep_projector(self, student_dim: int, teacher_dim: int) -> nn.Linear | None:
        if student_dim == teacher_dim:
            return None
        if self.rep_projector is None:
            self.rep_projector = nn.Linear(student_dim, teacher_dim, bias=False).to(get_device_id())
            if self.actor_optimizer is not None and not self._rep_projector_in_optimizer:
                self.actor_optimizer.add_param_group({"params": self.rep_projector.parameters()})
                self._rep_projector_in_optimizer = True
        return self.rep_projector

    def _get_or_create_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> LowRankCrossArchProjector | DirectLowRankCrossArchProjector:
        if self.low_rank_projector is None:
            rank = int(self.config.get("rep_low_rank", 256))
            checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            ps_type, mlp_mult, mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=checkpoint,
            )
            subspace_mode = "full"
            if checkpoint:
                subspace_mode = infer_subspace_mode_from_checkpoint(
                    load_preexp_checkpoint_dict(checkpoint)
                )
            if subspace_mode == "direct":
                if ps_type != "linear":
                    raise ValueError(
                        f"Direct preexp2 checkpoint at {checkpoint} requires linear P_S, got {ps_type!r}"
                    )
                self.low_rank_projector = DirectLowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                ).to(get_device_id())
            else:
                self.low_rank_projector = LowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                    ps_projector_type=ps_type,
                    mlp_hidden_mult=mlp_mult,
                    mlp_hidden_dim=mlp_hidden_dim,
                ).to(get_device_id())
            if checkpoint:
                self.low_rank_projector.load_from_preexp_checkpoint(checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.low_rank_projector.freeze_student_projectors()
                if self.low_rank_projector._loaded_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no P_S weights were loaded from checkpoint. "
                        "Rep loss will use random frozen P_S; set rep_low_rank_init_checkpoint."
                    )
            elif self.actor_optimizer is not None and not self._low_rank_projector_in_optimizer:
                trainable_params = self.low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._low_rank_projector_in_optimizer = True
        return self.low_rank_projector

    def _get_or_create_residual_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> ResidualLowRankCrossArchProjector:
        if self.residual_low_rank_projector is None:
            head_rank = int(self.config.get("rep_head_rank", 16))
            tail_rank = int(self.config.get("rep_low_rank", 16))
            head_checkpoint = self.config.get("rep_head_init_checkpoint", None)
            tail_checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            if not head_checkpoint:
                raise ValueError(
                    "rep_projector_mode=low_rank_residual requires rep_head_init_checkpoint "
                    "(pre-experiment 2 head ps_bank.pt, e.g. rank_16/ps_bank.pt)"
                )
            tail_ps_type, tail_mlp_mult, tail_mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=tail_checkpoint,
            )
            self.residual_low_rank_projector = ResidualLowRankCrossArchProjector(
                num_layers=num_layers,
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                head_rank=head_rank,
                tail_rank=tail_rank,
                num_teacher_layers=num_teacher_layers,
                tail_ps_projector_type=tail_ps_type,
                tail_mlp_hidden_mult=tail_mlp_mult,
                tail_mlp_hidden_dim=tail_mlp_hidden_dim,
            ).to(get_device_id())
            self.residual_low_rank_projector.load_head_from_preexp_checkpoint(head_checkpoint)
            if tail_checkpoint:
                self.residual_low_rank_projector.load_tail_from_preexp_checkpoint(tail_checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.residual_low_rank_projector.freeze_student_projectors()
                if self.residual_low_rank_projector._loaded_tail_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no tail P_S loaded. "
                        "Set rep_low_rank_init_checkpoint to a residual ps_tail_bank.pt."
                    )
            elif self.actor_optimizer is not None and not self._residual_low_rank_projector_in_optimizer:
                trainable_params = self.residual_low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._residual_low_rank_projector_in_optimizer = True
        return self.residual_low_rank_projector

    def _extract_response_hidden_repr_from_output(
        self,
        output,
        response_mask: torch.Tensor,
        *,
        indices=None,
        batch_size: int | None = None,
        seqlen: int | None = None,
        pad_size: int = 0,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> torch.Tensor:
        if rep_distillation_layers != "last":
            if self.use_remove_padding:
                return hidden_states_tuple_to_response_repr(
                    output.hidden_states,
                    response_mask,
                    rep_distillation_positions,
                    rep_distillation_layers,
                    indices=indices,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    use_ulysses_sp=self.use_ulysses_sp,
                    pad_size=pad_size,
                    use_remove_padding=True,
                    last_k=rep_distillation_last_k,
                    first_k=rep_distillation_first_k,
                )
            return extract_teacher_response_hidden_repr(
                output.hidden_states,
                response_mask,
                rep_distillation_positions,
                rep_distillation_layers,
                last_k=rep_distillation_last_k,
                first_k=rep_distillation_first_k,
            )

        if self.use_remove_padding:
            last_hidden_rmpad = _squeeze_hidden_rmpad(output.hidden_states[-1])
            if self.use_ulysses_sp:
                last_hidden_rmpad = gather_outputs_and_unpad(
                    last_hidden_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
            full_last_hidden = pad_input(
                hidden_states=last_hidden_rmpad,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            hidden_states = full_last_hidden.float()
        else:
            hidden_states = output.hidden_states[-1].float()

        return extract_teacher_response_hidden_repr(
            hidden_states,
            response_mask,
            rep_distillation_positions,
            "last",
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_hidden_repr(self, micro_batch) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        rep_distillation_positions = validate_rep_distillation_positions(
            self.config.get("rep_distillation_positions", "last")
        )
        rep_distillation_layers = validate_rep_distillation_layers(
            self.config.get("rep_distillation_layers", "last")
        )
        rep_distillation_last_k = int(self.config.get("rep_distillation_last_k", 32))
        rep_distillation_first_k = int(self.config.get("rep_distillation_first_k", 50))

        return forward_response_hidden_repr(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            use_remove_padding=self.use_remove_padding,
            use_ulysses_sp=self.use_ulysses_sp,
            ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
            multi_modal_inputs=multi_modal_inputs,
            positions=rep_distillation_positions,
            layers=rep_distillation_layers,
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_attn_rows(self, micro_batch) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        att_distillation_positions = validate_rep_distillation_positions(
            self.config.get("att_distillation_positions", "last")
        )
        att_distillation_layers = validate_rep_distillation_layers(
            self.config.get("att_distillation_layers", "last")
        )

        return forward_response_attn_rows(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            multi_modal_inputs=multi_modal_inputs,
            positions=att_distillation_positions,
            layers=att_distillation_layers,
            last_k=int(self.config.get("att_distillation_last_k", 32)),
            first_k=int(self.config.get("att_distillation_first_k", 50)),
            max_context_len=int(self.config.get("att_distillation_max_key_len", 4096)),
        )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        return_response_hidden_repr: bool = False,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
            response_hidden_repr: # (bs, response_len, hidden_dim) or None
        """
        response_length = micro_batch["responses"].size(-1)
        response_mask = None
        if return_response_hidden_repr:
            if "response_mask" in micro_batch:
                response_mask = micro_batch["response_mask"]
            else:
                response_mask = micro_batch["attention_mask"][:, -response_length:]
        response_hidden_repr = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                pad_size = 0
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        pad_size=pad_size,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # For top-k OPD, only sampled-token and top-k log-probabilities
                    # are consumed downstream.  Materialising log_softmax over the
                    # entire vocabulary is prohibitively expensive for long CoTs.
                    # Keep the per-token normalizer and gather only the required
                    # logits instead; this is exactly equivalent to log_softmax.
                    need_topk = top_k > 0
                    if need_topk:
                        log_normalizer = verl_F.logsumexp_over_vocab_with_chunking(logits_rmpad)
                        log_probs = logits_rmpad.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1) - log_normalizer
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        entropy_logits = logits_rmpad
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(entropy_logits)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        topk_log_probs = logits_rmpad.gather(dim=-1, index=topk_ids)
                        topk_log_probs = topk_log_probs - log_normalizer.unsqueeze(-1)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Top-k OPD only needs selected log-probabilities.  Avoid a
                    # dense [batch, response, vocab] log_softmax allocation.
                    need_topk = top_k > 0
                    if need_topk:
                        log_normalizer = verl_F.logsumexp_over_vocab_with_chunking(logits)
                        log_probs = logits.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1) - log_normalizer
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        # The chunked implementation expects [tokens, vocab];
                        # flattening response positions keeps its workspace bounded.
                        entropy_logits = logits.flatten(0, -2)
                        if not self.config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(entropy_logits).view(logits.shape[:-1])
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, entropy_logits
                            ).view(logits.shape[:-1])
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        topk_log_probs = logits.gather(dim=-1, index=topk_ids)
                        topk_log_probs = topk_log_probs - log_normalizer.unsqueeze(-1)

            return entropy, log_probs, topk_ids, topk_log_probs, response_hidden_repr

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self, *, diagnose_update_norm: bool = False):
        assert self.config.grad_clip is not None

        parameter_snapshots = None
        parameter_norm_sq = None
        if diagnose_update_norm:
            trainable_parameters = [parameter for parameter in self.actor_module.parameters() if parameter.requires_grad]
            parameter_snapshots = [parameter.detach().clone() for parameter in trainable_parameters]
            parameter_norm_sq = torch.zeros((), device=get_device_id(), dtype=torch.float32)
            for parameter in trainable_parameters:
                parameter_norm_sq += torch.sum(parameter.detach() * parameter.detach(), dtype=torch.float32)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()

        update_metrics = None
        if diagnose_update_norm:
            update_norm_sq = torch.zeros((), device=get_device_id(), dtype=torch.float32)
            for before, parameter in zip(parameter_snapshots, trainable_parameters, strict=True):
                delta = parameter.detach() - before
                update_norm_sq += torch.sum(delta * delta, dtype=torch.float32)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(parameter_norm_sq, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(update_norm_sq, op=torch.distributed.ReduceOp.SUM)
            parameter_norm = parameter_norm_sq.sqrt()
            update_norm = update_norm_sq.sqrt()
            update_metrics = {
                "actor/diagnostics_parameter_norm": parameter_norm.item(),
                "actor/diagnostics_update_norm": update_norm.item(),
                "actor/diagnostics_relative_update_norm": (
                    update_norm / parameter_norm.clamp_min(torch.finfo(parameter_norm.dtype).tiny)
                ).item(),
            }
        return grad_norm, update_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_layerwise_opd_support(self, data: DataProto) -> torch.Tensor:
        """Freeze self-support LayerwiseOPD ids at the rollout-policy snapshot.

        The result is ``[B,L,T,K]`` over the compact training positions.  It
        is computed before reward-model teacher labels and is reused unchanged
        by the subsequent actor update.
        """
        if not self.config.get("use_layerwise_opd_distillation", False):
            raise ValueError("layerwise support requested while LayerwiseOPD is disabled")
        if self.config.get("layerwise_opd_support", "final_policy") != "self_student":
            raise ValueError("compute_layerwise_opd_support requires layerwise_opd_support=self_student")
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        positions = validate_rep_distillation_positions(self.config.get("layerwise_opd_positions", "last_k"))
        last_k = int(self.config.get("layerwise_opd_last_k", 64))
        first_k = int(self.config.get("layerwise_opd_first_k", 64))
        top_k = int(self.config.get("layerwise_opd_top_k", 16))
        layers = parse_jalign_layers(self.config.get("layerwise_opd_student_layers", "8,14,20"))
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        data = data.select(
            batch_keys=["responses", "response_mask", "input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["multi_modal_inputs"] if has_multi_modal_inputs else [],
        )
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
        supports = []
        pair_mode = self.config.get("layerwise_opd_pair_selection", "none") != "none"
        snapshots = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                # top_k=1 avoids a dense full-vocab log_softmax in the ordinary
                # policy branch; the self-support readout below owns its exact K.
                _, _, _, _, hidden = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    top_k=1,
                    return_response_hidden_repr=True,
                    rep_distillation_positions=positions,
                    rep_distillation_layers="all",
                    rep_distillation_last_k=last_k,
                    rep_distillation_first_k=first_k,
                )
                # The all-layer helper has already removed HF's embedding slot,
                # so configured decoder-block indices address this tensor directly.
                selected = select_stacked_block_hidden_states(hidden, layers)
                fsdp_context = (
                    FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                    if isinstance(self.actor_module, FSDP)
                    else nullcontext()
                )
                with fsdp_context:
                    final_norm, lm_head = resolve_final_norm_and_lm_head(self.actor_module)
                    if pair_mode:
                        from verl.utils.transition_support import snapshot
                        ids, frozen_path = snapshot(hidden, layers, final_norm, lm_head,
                                                    int(self.config.get("layerwise_opd_token_chunk_size", 32)))
                        supports.append(ids)
                        snapshots.append(frozen_path)
                        continue
                    ids, _ = layerwise_topk_log_probs(
                        selected,
                        top_k=top_k,
                        final_norm=final_norm,
                        lm_head=lm_head,
                        token_chunk_size=int(self.config.get("layerwise_opd_token_chunk_size", 32)),
                        temperature=float(self.config.get("layerwise_opd_temperature", 1.0)),
                    )
            supports.append(ids)
            del hidden, selected
        support = torch.cat(supports, dim=0)
        if use_dynamic_bsz:
            support = restore_dynamic_batch(support, batch_idx_list)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pair_mode:
            frozen_path = torch.cat(snapshots, dim=0)
            if use_dynamic_bsz:
                frozen_path = restore_dynamic_batch(frozen_path, batch_idx_list)
            return support, frozen_path
        return support

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_jalign_opd_self_support(self, data: DataProto) -> torch.Tensor:
        """Freeze per-layer JOPD support using the current refreshed student lenses."""
        if not self.config.get("use_jlens_distillation", False):
            raise ValueError("JOPD self support requested while JAlign is disabled")
        if validate_jalign_loss_type(self.config.get("jalign_loss_type", "kl")) != "opd":
            raise ValueError("JOPD self support requires jalign_loss_type=opd")
        if self.config.get("jalign_opd_support", "final_policy") != "self_layerwise":
            raise ValueError("compute_jalign_opd_self_support requires jalign_opd_support=self_layerwise")
        self.actor_module.eval()
        artifacts = getattr(self, "_jalign_student_artifacts", None)
        if artifacts is None:
            artifacts = JLensArtifacts.load(
                self.config.get("student_jlens_checkpoint"),
                None,
                expected_layers=parse_jalign_layers(self.config.get("jalign_student_layers", "8,14,20")),
            )
            self._jalign_student_artifacts = artifacts
        temperature = data.meta_info["temperature"]
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        positions = validate_rep_distillation_positions(self.config.get("jalign_positions", "last_k"))
        last_k = int(self.config.get("jalign_last_k", 64))
        first_k = int(self.config.get("jalign_first_k", 64))
        top_k = int(self.config.get("layerwise_opd_top_k", 16))
        layers = parse_jalign_layers(self.config.get("jalign_student_layers", "8,14,20"))
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        data = data.select(
            batch_keys=["responses", "response_mask", "input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["multi_modal_inputs"] if has_multi_modal_inputs else [],
        )
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
        supports = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, _, _, _, hidden = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, top_k=1,
                    return_response_hidden_repr=True, rep_distillation_positions=positions,
                    rep_distillation_layers="all", rep_distillation_last_k=last_k,
                    rep_distillation_first_k=first_k,
                )
                selected = select_stacked_block_hidden_states(hidden, layers)
                transported = jlens_transported_residuals(selected, artifacts=artifacts)
                fsdp_context = FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False) if isinstance(self.actor_module, FSDP) else nullcontext()
                with fsdp_context:
                    final_norm, lm_head = resolve_final_norm_and_lm_head(self.actor_module)
                    ids, _ = layerwise_topk_log_probs(
                        transported, top_k=top_k, final_norm=final_norm, lm_head=lm_head,
                        temperature=float(self.config.get("jalign_temperature", 1.0)),
                    )
            supports.append(ids)
        support = torch.cat(supports, dim=0)
        return restore_dynamic_batch(support, batch_idx_list) if use_dynamic_bsz else support

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_rep_distillation = data.meta_info.get("use_rep_distillation", self.config.get("use_rep_distillation", False))
        use_jlens_distillation = self.config.get("use_jlens_distillation", False)
        use_direct_softmax_distillation = self.config.get("use_direct_softmax_distillation", False)
        use_layerwise_opd_distillation = self.config.get("use_layerwise_opd_distillation", False)
        if sum((use_rep_distillation, use_jlens_distillation, use_direct_softmax_distillation, use_layerwise_opd_distillation)) > 1:
            raise ValueError("representation, JAlign, DirectSoftmax, and LayerwiseOPD are mutually exclusive")
        jalign_positions = validate_rep_distillation_positions(self.config.get("jalign_positions", "last_k"))
        jalign_last_k = int(self.config.get("jalign_last_k", 64))
        jalign_first_k = int(self.config.get("jalign_first_k", 64))
        jalign_temperature = float(self.config.get("jalign_temperature", 1.0))
        jalign_coef = float(self.config.get("jalign_coef", 1.0))
        jalign_loss_type = validate_jalign_loss_type(self.config.get("jalign_loss_type", "kl"))
        jalign_opd_support = str(self.config.get("jalign_opd_support", "final_policy")).strip().lower()
        if jalign_opd_support not in {"final_policy", "self_layerwise"}:
            raise ValueError(f"unknown jalign_opd_support={jalign_opd_support!r}")
        jalign_student_transport = validate_jalign_student_transport(
            self.config.get("jalign_student_transport", "frozen")
        )
        jalign_gradient_routing = validate_jalign_gradient_routing(
            self.config.get("jalign_gradient_routing", "jacobian")
        )
        jalign_opd_aggregation = str(
            self.config.get("jalign_opd_aggregation", "equal_weight")
        ).strip().lower()
        if jalign_opd_aggregation not in {"equal_weight", "legacy_additive"}:
            raise ValueError(
                "jalign_opd_aggregation must be 'equal_weight' or 'legacy_additive', "
                f"got {jalign_opd_aggregation!r}"
            )
        jflow_coef = float(self.config.get("jflow_coef", 1.0))
        jflow_token_offset = int(self.config.get("jflow_token_offset", 8))
        if jflow_coef < 0:
            raise ValueError("jflow_coef must be non-negative")
        if jflow_token_offset <= 0:
            raise ValueError("jflow_token_offset must be positive")
        if jalign_loss_type == "flow_opd" and jalign_opd_support != "final_policy":
            raise ValueError("J-Flow currently requires jalign_opd_support=final_policy")
        if jalign_loss_type == "flow_opd" and jalign_gradient_routing != "identity_credit":
            raise ValueError("J-Flow formal mode requires jalign_gradient_routing=identity_credit")
        if jalign_gradient_routing == "identity_credit":
            if jalign_loss_type not in {"opd", "flow_opd"}:
                raise ValueError(
                    "JLens identity credit is defined only for jalign_loss_type=opd or flow_opd"
                )
            if jalign_student_transport not in {"frozen", "periodic_refresh"}:
                raise ValueError(
                    "JLens identity credit requires jalign_student_transport=frozen "
                    "or periodic_refresh"
                )
        jalign_stoch_rank = int(self.config.get("jalign_stoch_rank", 8))
        jalign_stoch_eps = float(self.config.get("jalign_stoch_eps", 1e-3))
        jalign_stoch_probe = validate_jalign_stoch_probe(self.config.get("jalign_stoch_probe", "rademacher"))
        jalign_student_layers = resolve_jalign_layers(
            self.config.get("jalign_student_layers", "8,14,20"),
            lens_checkpoint=self.config.get("student_jlens_checkpoint"),
        )
        direct_softmax_positions = validate_rep_distillation_positions(
            self.config.get("direct_softmax_positions", "last_k")
        )
        direct_softmax_last_k = int(self.config.get("direct_softmax_last_k", 64))
        direct_softmax_first_k = int(self.config.get("direct_softmax_first_k", 64))
        direct_softmax_temperature = float(self.config.get("direct_softmax_temperature", 1.0))
        direct_softmax_coef = float(self.config.get("direct_softmax_coef", 1.0))
        direct_softmax_student_layers = parse_jalign_layers(
            self.config.get("direct_softmax_student_layers", "8,14,20")
        )
        layerwise_opd_positions = validate_rep_distillation_positions(
            self.config.get("layerwise_opd_positions", "last_k")
        )
        layerwise_opd_last_k = int(self.config.get("layerwise_opd_last_k", 64))
        layerwise_opd_first_k = int(self.config.get("layerwise_opd_first_k", 64))
        layerwise_opd_temperature = float(self.config.get("layerwise_opd_temperature", 1.0))
        layerwise_opd_token_chunk_size = int(self.config.get("layerwise_opd_token_chunk_size", 32))
        layerwise_opd_coef = float(self.config.get("layerwise_opd_coef", 1.0))
        layerwise_opd_aggregation = str(
            self.config.get("layerwise_opd_aggregation", "equal_weight")
        ).strip().lower()
        if layerwise_opd_aggregation not in {"equal_weight", "legacy_additive"}:
            raise ValueError(f"unknown layerwise_opd_aggregation={layerwise_opd_aggregation!r}")
        layerwise_opd_top_k = int(self.config.get("layerwise_opd_top_k", 16))
        pair_selection = self.config.get("layerwise_opd_pair_selection", "none")
        pair_mode = pair_selection != "none"
        if pair_mode and (pair_selection not in {"max", "min"} or
                         self.config.get("layerwise_opd_objective") != "depth_transition" or
                         self.config.get("layerwise_opd_support") != "self_student" or
                         self.config.get("layerwise_opd_adaptive", False) or
                         self.config.get("layerwise_opd_sample_one_layer", False)):
            raise ValueError("pair selection requires nonadaptive depth_transition/self_student")
        layerwise_objective = str(self.config.get("layerwise_opd_objective", "absolute")).strip().lower()
        transition_weights_spec = str(self.config.get("layerwise_opd_transition_weights", "")).strip()
        if transition_weights_spec and (layerwise_objective != "depth_transition" or pair_mode or bool(self.config.get("layerwise_opd_adaptive", False))):
            raise ValueError("fixed transition weights require nonadaptive depth_transition without pair selection")
        transition_weights = _fixed_transition_weights(transition_weights_spec, 3) if transition_weights_spec else None
        delta_align = bool(self.config.get("layerwise_opd_delta_align", False))
        reverse_transition_kl = bool(self.config.get("layerwise_opd_reverse_transition_kl", False))
        weight_only = bool(self.config.get("layerwise_opd_weight_only", False))
        center_only = bool(self.config.get("layerwise_opd_center_only", False))
        if sum((reverse_transition_kl, weight_only, center_only)) > 1:
            raise ValueError("transition ablation modes are mutually exclusive")
        if reverse_transition_kl and (
            layerwise_objective != "depth_transition" or pair_mode or delta_align
            or transition_weights is not None or self.config.get("layerwise_opd_adaptive", False)
            or self.config.get("layerwise_opd_sample_one_layer", False)
        ):
            raise ValueError("reverse transition KL requires fixed nonadaptive depth transitions")
        if delta_align and (layerwise_objective != "depth_transition" or pair_mode or
                            transition_weights is not None or
                            self.config.get("layerwise_opd_adaptive", False) or
                            self.config.get("layerwise_opd_sample_one_layer", False)):
            raise ValueError("delta alignment requires fixed nonadaptive depth transitions")
        layerwise_opd_detach_lm_head = bool(self.config.get("layerwise_opd_detach_lm_head", False))
        layerwise_opd_support = str(self.config.get("layerwise_opd_support", "final_policy")).strip().lower()
        if layerwise_opd_support not in {"final_policy", "self_student", "teacher_layerwise"}:
            raise ValueError(f"unknown layerwise_opd_support={layerwise_opd_support!r}")
        layerwise_opd_teacher_target = str(
            self.config.get("layerwise_opd_teacher_target", "layerwise")
        ).strip().lower()
        if layerwise_opd_support == "teacher_layerwise" and layerwise_opd_teacher_target != "layerwise":
            raise ValueError(
                "teacher_layerwise support requires layerwise_opd_teacher_target=layerwise"
            )
        layerwise_opd_student_layers = parse_jalign_layers(
            self.config.get("layerwise_opd_student_layers", "8,14,20")
        )
        layerwise_opd_jacobian_backward = bool(self.config.get("layerwise_opd_jacobian_backward", False))
        layerwise_opd_gradient_routing = resolve_layerwise_opd_gradient_routing(
            self.config.get("layerwise_opd_gradient_routing", "raw"),
            legacy_frozen_jacobian_backward=layerwise_opd_jacobian_backward,
        )
        layerwise_opd_readout = str(self.config.get("layerwise_opd_readout", "raw")).strip().lower()
        if layerwise_opd_readout not in {"raw", "frozen_affine"}:
            raise ValueError(f"unknown layerwise_opd_readout={layerwise_opd_readout!r}")
        layerwise_opd_diagnostics_enabled = bool(self.config.get("layerwise_opd_diagnostics", False))
        layerwise_opd_diagnostics_interval = int(self.config.get("layerwise_opd_diagnostics_interval", 25))
        online_jcr_norm_chunk_max_tokens = int(
            self.config.get("layerwise_opd_online_jcr_norm_chunk_max_tokens", 0)
        )
        if online_jcr_norm_chunk_max_tokens < 0:
            raise ValueError("Online-JCR norm chunk max tokens must be non-negative")
        if layerwise_opd_diagnostics_enabled and not use_layerwise_opd_distillation:
            raise ValueError("layerwise_opd_diagnostics requires LayerwiseOPD")
        global_step = int(data.meta_info.get("global_steps", -1))
        layerwise_opd_diagnostics = should_run_layerwise_opd_diagnostics(
            enabled=layerwise_opd_diagnostics_enabled,
            interval=layerwise_opd_diagnostics_interval,
            global_step=global_step,
        )
        layerwise_opd_sample_one_layer = bool(
            data.meta_info.get(
                "layerwise_opd_sample_one_layer",
                self.config.get("layerwise_opd_sample_one_layer", False),
            )
        )
        validate_layerwise_opd_sample_one_layer(
            layerwise_opd_sample_one_layer,
            gradient_routing=layerwise_opd_gradient_routing,
            aggregation=layerwise_opd_aggregation,
        )
        sampled_adjacent = self.config.get("layerwise_opd_objective", "absolute") == "sampled_adjacent_transition"
        if sampled_adjacent and (not layerwise_opd_sample_one_layer or layerwise_opd_support != "final_policy" or self.config.get("layerwise_opd_adaptive", False)):
            raise ValueError("sampled adjacent transitions require sampled raw final-policy support without adaptive weights")
        if layerwise_opd_sample_one_layer and layerwise_opd_readout != "raw":
            raise ValueError("sample-one-layer LayerwiseOPD currently requires layerwise_opd_readout=raw")
        if layerwise_opd_gradient_routing in {"online_jcr", "online_jcr_norm"} and layerwise_opd_support != "final_policy":
            raise ValueError("Online-JCR currently requires layerwise_opd_support=final_policy")
        layerwise_opd_jacobian_artifacts = None
        if layerwise_opd_gradient_routing == "frozen_jlens":
            layerwise_opd_jlens_checkpoint = self.config.get("layerwise_opd_student_jlens_checkpoint", None)
            if layerwise_opd_jlens_checkpoint in (None, "", "null"):
                raise ValueError(
                    "frozen_jlens LayerwiseOPD routing requires layerwise_opd_student_jlens_checkpoint"
                )
            layerwise_opd_jacobian_artifacts = self._layerwise_opd_jacobian_artifacts
            if layerwise_opd_jacobian_artifacts is None:
                layerwise_opd_jacobian_artifacts = JLensArtifacts.load(
                    layerwise_opd_jlens_checkpoint,
                    None,
                    expected_layers=layerwise_opd_student_layers,
                )
                self._layerwise_opd_jacobian_artifacts = layerwise_opd_jacobian_artifacts
        layerwise_opd_affine_artifacts = None
        if layerwise_opd_readout == "frozen_affine":
            checkpoint = self.config.get("layerwise_opd_student_affine_checkpoint", None)
            if checkpoint in (None, "", "null"):
                raise ValueError(
                    "frozen_affine LayerwiseOPD readout requires "
                    "layerwise_opd_student_affine_checkpoint"
                )
            layerwise_opd_affine_artifacts = self._layerwise_opd_affine_artifacts
            if layerwise_opd_affine_artifacts is None:
                layerwise_opd_affine_artifacts = FrozenAffineReadoutArtifacts.load(
                    checkpoint, expected_layers=layerwise_opd_student_layers
                )
                self._layerwise_opd_affine_artifacts = layerwise_opd_affine_artifacts
        use_att_distillation = self.config.get("use_att_distillation", False)
        rep_distillation_only = self.config.get("rep_distillation_only", False)
        pure_forward_kl = self.config.get("pure_forward_kl", False)
        if pure_forward_kl:
            assert rep_distillation_only and use_direct_softmax_distillation
            assert not self.config.use_kl_loss and float(self.config.entropy_coeff) == 0.0
            assert self.config.get("pure_forward_kl_readout") in {"raw", "jlens", "final_only"}
            assert not use_jlens_distillation and not use_layerwise_opd_distillation
            assert not use_rep_distillation and not self.config.get("use_att_distillation", False)
        rep_distillation_coef = self.config.get("rep_distillation_coef", 1.0)
        rep_distillation_crossfade_steps = int(self.config.get("rep_distillation_crossfade_steps", 0))
        if rep_distillation_crossfade_steps < 0:
            raise ValueError("rep_distillation_crossfade_steps must be non-negative")
        rep_distillation_effective_coef = float(rep_distillation_coef)
        if rep_distillation_crossfade_steps > 0 and global_step >= 0:
            rep_distillation_effective_coef *= max(
                0.0, 1.0 - float(global_step - 1) / float(rep_distillation_crossfade_steps)
            )
        if rep_distillation_crossfade_steps > 0:
            if rep_distillation_only or self.config.get("rep_distillation_layers") != "last":
                raise ValueError("LastOPD requires joint token supervision and last-layer representations")
        att_distillation_coef = self.config.get("att_distillation_coef", 1.0)
        att_distillation_loss_type = validate_att_distillation_loss_type(
            self.config.get("att_distillation_loss", "kl")
        )
        att_distillation_temperature = float(self.config.get("att_distillation_temperature", 1.0))
        att_distillation_positions = validate_rep_distillation_positions(
            self.config.get("att_distillation_positions", "last")
        )
        att_distillation_layers = validate_rep_distillation_layers(
            self.config.get("att_distillation_layers", "last")
        )
        att_distillation_last_k = int(self.config.get("att_distillation_last_k", 32))
        att_distillation_first_k = int(self.config.get("att_distillation_first_k", 50))
        multi_layer_att = att_distillation_layers != "last"
        rep_distillation_positions = validate_rep_distillation_positions(
            self.config.get("rep_distillation_positions", "last")
        )
        rep_distillation_layers = validate_rep_distillation_layers(
            self.config.get("rep_distillation_layers", "last")
        )
        rep_distillation_last_k = int(self.config.get("rep_distillation_last_k", 32))
        rep_distillation_first_k = int(self.config.get("rep_distillation_first_k", 50))
        multi_layer_rep = rep_distillation_layers != "last"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        if not rep_distillation_only:
            select_keys.extend(["old_log_probs", "advantages"])
        if use_rep_distillation:
            select_keys.append("teacher_last_hidden_repr")
        if use_jlens_distillation:
            if jalign_loss_type == "mse":
                select_keys.append("teacher_jvectors")
            elif jalign_loss_type == "opd":
                select_keys.append("teacher_jalign_candidate_log_probs")
                if jalign_opd_support == "self_layerwise":
                    select_keys.append("jalign_student_top_k_ids")
            elif jalign_loss_type == "flow_opd":
                select_keys.extend([
                    "teacher_jalign_candidate_log_probs",
                    "teacher_jflow_anchor_candidate_log_probs",
                    "teacher_jflow_raw_candidate_log_probs",
                ])
            else:
                select_keys.append("teacher_jlogits")
        if use_direct_softmax_distillation:
            select_keys.append("teacher_pure_fkl_hidden" if pure_forward_kl else "teacher_direct_logits")
        if use_layerwise_opd_distillation:
            select_keys.append("teacher_layerwise_candidate_log_probs")
            if self.config.get("expertalign_layerwise_per_layer", False):
                select_keys.extend([
                    "expertalign_teacher_layerwise_0",
                    "expertalign_teacher_layerwise_1",
                    "expertalign_teacher_layerwise_2",
                    "expertalign_base_layerwise_candidate_log_probs",
                ])
            if "expertalign_layerwise_mask" in data.batch:
                select_keys.append("expertalign_layerwise_mask")
            if pair_mode:
                select_keys.append("transition_snapshot_log_probs")
            if layerwise_opd_sample_one_layer:
                select_keys.append("layerwise_opd_sampled_slot")
            if layerwise_opd_support == "self_student":
                select_keys.append("layerwise_student_top_k_ids")
            elif layerwise_opd_support == "teacher_layerwise":
                select_keys.append("teacher_layerwise_top_k_ids")
        if use_att_distillation:
            select_keys.append("teacher_attn_rows")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")
        if "teacher_on_student_log_probs" in data.batch.keys():
            select_keys.append("teacher_on_student_log_probs")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if "data_source" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("data_source")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                online_jcr_layer_slot = None
                layerwise_sampled_layer_slot = None
                if use_layerwise_opd_distillation and layerwise_opd_sample_one_layer:
                    sampled_slots = mini_batch.batch["layerwise_opd_sampled_slot"]
                    unique_sampled_slots = torch.unique(sampled_slots).detach().cpu()
                    if unique_sampled_slots.numel() != 1:
                        raise ValueError(
                            "each PPO minibatch must contain exactly one sampled LayerwiseOPD slot, "
                            f"got {unique_sampled_slots.tolist()}"
                        )
                    layerwise_sampled_layer_slot = int(unique_sampled_slots.item())
                    if not 0 <= layerwise_sampled_layer_slot < len(layerwise_opd_student_layers):
                        raise ValueError(
                            f"sampled LayerwiseOPD slot {layerwise_sampled_layer_slot} is outside "
                            f"[0, {len(layerwise_opd_student_layers) - 1}]"
                        )
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        synchronized_slot = torch.tensor(
                            [
                                layerwise_sampled_layer_slot
                                if torch.distributed.get_rank() == 0
                                else 0
                            ],
                            device=get_device_id(),
                            dtype=torch.long,
                        )
                        torch.distributed.broadcast(synchronized_slot, src=0)
                        synchronized_slot_value = int(synchronized_slot.item())
                        if layerwise_sampled_layer_slot != synchronized_slot_value:
                            raise RuntimeError(
                                "sampled LayerwiseOPD slot differs across distributed actor ranks: "
                                f"local={layerwise_sampled_layer_slot}, rank0={synchronized_slot_value}"
                            )
                if layerwise_opd_gradient_routing in {"online_jcr", "online_jcr_norm", "online_jcr_activation"}:
                    online_jcr_layer_slot = self._online_jcr_layer_cursor % len(layerwise_opd_student_layers)
                    self._online_jcr_layer_cursor += 1
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        synchronized_slot = torch.tensor(
                            [online_jcr_layer_slot], device=get_device_id(), dtype=torch.long
                        )
                        torch.distributed.broadcast(synchronized_slot, src=0)
                        online_jcr_layer_slot = int(synchronized_slot.item())
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    # Keep full-vocab teacher J-/direct-logits on host; streamed KL
                    # pulls token chunks to GPU on demand (avoids a second [B,L,T,V]).
                    host_full_vocab_logits = {}
                    for host_key in ("teacher_jlogits", "teacher_direct_logits"):
                        if host_key in micro_batch.batch.keys():
                            host_full_vocab_logits[host_key] = micro_batch.batch.pop(host_key)
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, **host_full_vocab_logits}
                    if self.config.get("trim_common_padding", False):
                        if not pair_mode or self.use_remove_padding:
                            raise ValueError("common-padding trim is qualified for padded transition-pair updates")
                        from verl.utils.padded_microbatch import trim_common_padding
                        padded_width = model_inputs["input_ids"].shape[1]
                        model_inputs = trim_common_padding(model_inputs)
                        micro_batch_metrics["actor/padded_width_before_trim"] = float(padded_width)
                        micro_batch_metrics["actor/padded_width_after_trim"] = float(model_inputs["input_ids"].shape[1])
                    response_mask = model_inputs["response_mask"]
                    # PG (varlen) + attention distillation (eager padded) share one
                    # backward, so attention distillation still needs checkpointing
                    # disabled. Activation-only Online-JCR uses non-reentrant
                    # checkpointing for both forwards to bound the suffix-VJP peak.
                    grad_ckpt_ctx = (
                        disable_gradient_checkpointing(self.actor_module)
                        if use_att_distillation
                        else nullcontext()
                    )
                    old_log_prob = model_inputs.get("old_log_probs")
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    policy_loss = None
                    student_repr_from_forward = None
                    att_loss = None
                    with grad_ckpt_ctx:
                        online_jcr_routed_cotangent = None
                        online_jcr_norm_scale = None
                        online_jcr_raw_cotangent_norm = None
                        online_jcr_routed_cotangent_norm = None
                        if layerwise_opd_gradient_routing == "online_jcr_norm":
                            if layerwise_opd_support != "final_policy":
                                raise ValueError("norm-matched Online-JCR requires final-policy support")
                            sampled_layer_id = layerwise_opd_student_layers[online_jcr_layer_slot]
                            norm_chunk_indices = _online_jcr_norm_chunk_indices(
                                model_inputs["attention_mask"],
                                online_jcr_norm_chunk_max_tokens,
                            )
                            raw_cotangent_sq = torch.zeros((), device=get_device_id(), dtype=torch.float32)
                            routed_cotangent_sq = torch.zeros((), device=get_device_id(), dtype=torch.float32)
                            norm_chunk_token_counts = []
                            batch_size = int(response_mask.shape[0])
                            full_selected_tokens = build_compact_rep_distillation_position_mask(
                                response_mask,
                                layerwise_opd_positions,
                                layerwise_opd_last_k,
                                layerwise_opd_first_k,
                            ).sum().clamp_min(1)
                            for chunk_indices in norm_chunk_indices:
                                padding_chunk = len(chunk_indices) == 0
                                execution_indices = [0] if padding_chunk else chunk_indices
                                if len(norm_chunk_indices) == 1:
                                    # Preserve the legacy path byte-for-byte when no
                                    # split is needed; in particular, avoid introducing
                                    # indexed tensor views around FSDP checkpointing.
                                    chunk_inputs = model_inputs
                                    chunk_response_mask = response_mask
                                else:
                                    chunk_inputs = _slice_online_jcr_model_inputs(
                                        model_inputs,
                                        execution_indices,
                                        batch_size,
                                    )
                                    chunk_response_mask = response_mask.index_select(
                                        0,
                                        torch.tensor(execution_indices, device=response_mask.device, dtype=torch.long),
                                    )
                                raw_cotangent, routed_cotangent = _compute_online_jcr_cotangents(
                                    self.actor_module,
                                    chunk_inputs,
                                    chunk_response_mask,
                                    sampled_layer_id=sampled_layer_id,
                                    teacher_layer_slot=online_jcr_layer_slot,
                                    positions=layerwise_opd_positions,
                                    last_k=layerwise_opd_last_k,
                                    first_k=layerwise_opd_first_k,
                                    temperature=layerwise_opd_temperature,
                                    use_remove_padding=self.use_remove_padding,
                                    use_ulysses_sp=self.use_ulysses_sp,
                                    ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
                                )
                                if padding_chunk:
                                    del raw_cotangent, routed_cotangent
                                    continue
                                chunk_selected_tokens = build_compact_rep_distillation_position_mask(
                                    chunk_response_mask,
                                    layerwise_opd_positions,
                                    layerwise_opd_last_k,
                                    layerwise_opd_first_k,
                                ).sum()
                                # Each chunk loss is locally token-mean normalized. Restore
                                # the full-micro-batch token mean before accumulating norms.
                                chunk_loss_weight = chunk_selected_tokens / full_selected_tokens
                                raw_cotangent_sq += (raw_cotangent.float() * chunk_loss_weight).square().sum()
                                routed_cotangent_sq += (routed_cotangent.float() * chunk_loss_weight).square().sum()
                                norm_chunk_token_counts.append(
                                    int(chunk_inputs["attention_mask"].sum().item())
                                )
                                del raw_cotangent, routed_cotangent
                            online_jcr_raw_cotangent_norm = raw_cotangent_sq.sqrt()
                            online_jcr_routed_cotangent_norm = routed_cotangent_sq.sqrt()
                            online_jcr_norm_scale = torch.where(
                                online_jcr_routed_cotangent_norm > 0,
                                online_jcr_raw_cotangent_norm / online_jcr_routed_cotangent_norm,
                                torch.ones_like(online_jcr_routed_cotangent_norm),
                            ).detach()
                            micro_batch_metrics["actor/online_jcr_norm_chunk_count"] = float(
                                sum(bool(indices) for indices in norm_chunk_indices)
                            )
                            micro_batch_metrics["actor/online_jcr_norm_vjp_call_count"] = float(
                                len(norm_chunk_indices)
                            )
                            micro_batch_metrics["actor/online_jcr_norm_chunk_max_tokens"] = float(
                                max(norm_chunk_token_counts, default=0)
                            )
                            micro_batch_metrics["actor/online_jcr_norm_chunk_total_tokens"] = float(
                                sum(norm_chunk_token_counts)
                            )
                        if layerwise_opd_gradient_routing == "online_jcr_activation":
                            if layerwise_opd_support != "final_policy":
                                raise ValueError(
                                    "activation-only Online-JCR requires final-policy support"
                                )
                            sampled_layer_id = layerwise_opd_student_layers[online_jcr_layer_slot]
                            online_jcr_routed_cotangent = _compute_activation_only_online_jcr_cotangent(
                                self.actor_module,
                                model_inputs,
                                response_mask,
                                sampled_layer_id=sampled_layer_id,
                                teacher_layer_slot=online_jcr_layer_slot,
                                positions=layerwise_opd_positions,
                                last_k=layerwise_opd_last_k,
                                first_k=layerwise_opd_first_k,
                                temperature=layerwise_opd_temperature,
                                use_remove_padding=self.use_remove_padding,
                                use_ulysses_sp=self.use_ulysses_sp,
                                ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
                            )

                        if not rep_distillation_only:
                            # all return: (bsz, response_length)
                            calculate_entropy = False
                            if entropy_coeff != 0:
                                calculate_entropy = True

                            # Reuse the same forward for rep loss when combined with OPD (avoids a
                            # second full forward that breaks activation offload).
                            return_hidden = use_rep_distillation

                            # Check if we have 3D advantages (top-k sampling case)
                            # If so, we need to recompute top-k log probs for correct gradient
                            if advantages.dim() == 3:
                                top_k = advantages.shape[-1]
                                # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                                student_top_k_ids = None
                                if "union_top_k_ids" in model_inputs:
                                    student_top_k_ids = model_inputs["union_top_k_ids"]
                                elif "student_top_k_ids" in model_inputs:
                                    student_top_k_ids = model_inputs["student_top_k_ids"]

                                entropy, _, _, topk_log_probs, student_repr_from_forward = self._forward_micro_batch(
                                    model_inputs,
                                    temperature=temperature,
                                    calculate_entropy=calculate_entropy,
                                    top_k=top_k,
                                    student_top_k_ids=student_top_k_ids,
                                    return_response_hidden_repr=return_hidden,
                                    rep_distillation_positions=rep_distillation_positions,
                                    rep_distillation_layers=rep_distillation_layers,
                                    rep_distillation_last_k=rep_distillation_last_k,
                                    rep_distillation_first_k=rep_distillation_first_k,
                                )
                                log_prob_for_loss = topk_log_probs

                            else:
                                _, log_prob, _, _, student_repr_from_forward = self._forward_micro_batch(
                                    model_inputs,
                                    temperature=temperature,
                                    calculate_entropy=calculate_entropy,
                                    return_response_hidden_repr=return_hidden,
                                    rep_distillation_positions=rep_distillation_positions,
                                    rep_distillation_layers=rep_distillation_layers,
                                    rep_distillation_last_k=rep_distillation_last_k,
                                    rep_distillation_first_k=rep_distillation_first_k,
                                )
                                log_prob_for_loss = log_prob

                            format_mask = None
                            if "format_mask" in model_inputs.keys():
                                format_mask = model_inputs["format_mask"]

                            # for fully_async_policy recipe
                            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                                old_log_prob = model_inputs["old_log_probs"]
                            else:
                                if on_policy:
                                    print("on_policy")
                                    # For on-policy (ppo_epochs=1), use current policy as "old"
                                    # log_prob_for_loss is already 3D for top-k case
                                    old_log_prob = log_prob_for_loss.detach()
                                else:
                                    print("off_policy")
                                    # For off-policy, use stored log probs
                                    # For 3D top-k case, use stored log probs (union or student)
                                    if advantages.dim() == 3:
                                        if "union_top_k_log_probs" in model_inputs:
                                            old_log_prob = model_inputs["union_top_k_log_probs"]
                                        elif "student_top_k_log_probs" in model_inputs:
                                            old_log_prob = model_inputs["student_top_k_log_probs"]
                                        else:
                                            old_log_prob = model_inputs["old_log_probs"]
                                    else:
                                        old_log_prob = model_inputs["old_log_probs"]

                            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                            # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                            # Extract pre-computed rollout correction weights if present
                            # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                            rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                            # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                            # are computed centrally in ray_trainer.py for consistency and efficiency.
                            # This ensures metrics are computed uniformly across all batches at the trainer level
                            # and avoids redundant computation across workers and micro-batches.

                            # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                            # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                            policy_loss_fn = get_policy_loss_fn(loss_mode)

                            domain_is_code = None
                            if loss_agg_mode == "domain-equal-token-mean":
                                domain_is_code = domain_is_code_from_sources(
                                    model_inputs.get("data_source"),
                                    batch_size=response_mask.shape[0],
                                    device=response_mask.device,
                                )

                            policy_loss_kwargs = dict(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                                format_mask=format_mask,
                            )
                            if domain_is_code is not None:
                                policy_loss_kwargs["domain_is_code"] = domain_is_code
                            pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kwargs)
                            micro_batch_metrics.update(pg_metrics)

                            if entropy_coeff != 0:
                                entropy_loss = agg_loss(
                                    loss_mat=entropy,
                                    loss_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    domain_is_code=domain_is_code,
                                )

                                # compute policy loss
                                policy_loss = pg_loss - entropy_loss * entropy_coeff
                            else:
                                policy_loss = pg_loss

                            if self.config.use_kl_loss:
                                ref_log_prob = model_inputs["ref_log_prob"]
                                # compute kl loss
                                kld = kl_penalty(
                                    logprob=log_prob_for_loss,
                                    ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type,
                                )
                                kl_loss = agg_loss(
                                    loss_mat=kld,
                                    loss_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    domain_is_code=domain_is_code,
                                )

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                            micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor

                            if rep_distillation_crossfade_steps > 0:
                                token_weight = min(1.0, max(0.0, float(global_step - 1) / rep_distillation_crossfade_steps))
                                policy_loss = policy_loss * token_weight
                                micro_batch_metrics["actor/lastopd_token_weight"] = token_weight

                        if use_rep_distillation:
                            teacher_repr = model_inputs["teacher_last_hidden_repr"]
                            response_mask = model_inputs["response_mask"]
                            if student_repr_from_forward is not None:
                                student_repr = student_repr_from_forward
                            else:
                                student_repr = self._forward_response_hidden_repr(model_inputs)

                            num_rep_layers = student_repr.size(1) if multi_layer_rep else None
                            position_mask = None

                            if rep_distillation_positions == "last":
                                if multi_layer_rep:
                                    if student_repr.dim() != 3:
                                        raise ValueError(
                                            "Expected student multi-layer last-token repr (B, L, D)"
                                        )
                                else:
                                    if student_repr.dim() != 3:
                                        raise ValueError(
                                            "Expected student response hidden states (B, T, D) for last-token rep distillation"
                                        )
                                    last_indices = response_mask.long().sum(dim=1) - 1
                                    last_indices = last_indices.clamp(min=0)
                                    batch_indices = torch.arange(student_repr.size(0), device=student_repr.device)
                                    student_repr = student_repr[batch_indices, last_indices]
                            elif rep_distillation_positions in ("first_k", "last_k"):
                                if student_repr.shape != teacher_repr.shape:
                                    student_repr, _ = compact_response_repr_by_positions(
                                        student_repr,
                                        response_mask,
                                        rep_distillation_positions,
                                        rep_distillation_last_k,
                                        rep_distillation_first_k,
                                    )
                                position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    rep_distillation_positions,
                                    rep_distillation_last_k,
                                    rep_distillation_first_k,
                                )
                                if multi_layer_rep:
                                    if student_repr.dim() != 4 or teacher_repr.dim() != 4:
                                        raise ValueError(
                                            "Expected compact multi-layer repr (B, L, k, D) for first_k/last_k"
                                        )
                                elif student_repr.dim() != 3 or teacher_repr.dim() != 3:
                                    raise ValueError(
                                        "Expected compact single-layer repr (B, k, D) for first_k/last_k"
                                    )
                            else:
                                position_mask = response_mask
                                if multi_layer_rep:
                                    if student_repr.dim() != 4:
                                        raise ValueError(
                                            "Expected student multi-layer per-token repr (B, L, T, D)"
                                        )
                                elif student_repr.dim() != 3:
                                    raise ValueError(
                                        "teacher_last_hidden_repr is per-token but student repr is not (B, T, D)"
                                    )

                            student_dim = int(student_repr.size(-1))
                            teacher_dim = int(teacher_repr.size(-1))
                            micro_batch_metrics["rep/student_hidden_dim"] = float(student_dim)
                            micro_batch_metrics["rep/teacher_hidden_dim"] = float(teacher_dim)
                            if multi_layer_rep:
                                micro_batch_metrics["rep/num_student_layers"] = float(student_repr.size(1))
                                micro_batch_metrics["rep/num_teacher_layers_raw"] = float(teacher_repr.size(1))
                            rep_projector_mode = validate_rep_projector_mode(
                                self.config.get("rep_projector_mode", "full")
                            )
                            if student_dim == teacher_dim:
                                rep_projector_mode = "full"

                            if rep_projector_mode == "low_rank":
                                num_teacher_layers = teacher_repr.size(1) if multi_layer_rep else 1
                                low_rank_layers = num_rep_layers or 1
                                low_rank_projector = self._get_or_create_low_rank_projector(
                                    student_dim,
                                    teacher_dim,
                                    num_layers=low_rank_layers,
                                    num_teacher_layers=num_teacher_layers,
                                )
                                low_rank_projector.maybe_init_teacher_pca_from_batch(
                                    teacher_repr, num_layers=low_rank_layers
                                )
                                student_repr, teacher_repr = low_rank_projector.project_pair(
                                    student_repr,
                                    teacher_repr,
                                    num_layers=num_rep_layers,
                                )
                                micro_batch_metrics["rep/use_low_rank_projector"] = 1.0
                                micro_batch_metrics["rep/low_rank"] = float(low_rank_projector.rank)
                                micro_batch_metrics.update(low_rank_projector.projector_param_metrics())
                            elif rep_projector_mode == "low_rank_residual":
                                num_teacher_layers = teacher_repr.size(1) if multi_layer_rep else 1
                                low_rank_layers = num_rep_layers or 1
                                residual_projector = self._get_or_create_residual_low_rank_projector(
                                    student_dim,
                                    teacher_dim,
                                    num_layers=low_rank_layers,
                                    num_teacher_layers=num_teacher_layers,
                                )
                                micro_batch_metrics.update(
                                    residual_projector.head_alignment_metrics(
                                        student_repr,
                                        teacher_repr,
                                        num_layers=num_rep_layers,
                                    )
                                )
                                residual_projector.maybe_init_tail_pca_from_batch(
                                    teacher_repr, num_layers=low_rank_layers
                                )
                                student_repr, teacher_repr = residual_projector.project_pair(
                                    student_repr,
                                    teacher_repr,
                                    num_layers=num_rep_layers,
                                )
                                micro_batch_metrics["rep/use_low_rank_projector"] = 1.0
                                micro_batch_metrics["rep/low_rank"] = float(residual_projector.tail_rank)
                                micro_batch_metrics.update(residual_projector.projector_param_metrics())
                            else:
                                projector = self._get_or_create_rep_projector(student_dim, teacher_dim)
                                if projector is not None:
                                    student_repr = projector(student_repr)
                                    w = projector.weight.detach().float()
                                    micro_batch_metrics["rep/ps_weight_norm_mean"] = float(w.norm().item())
                                    micro_batch_metrics["rep/ps_weight_norm_max"] = float(w.norm().item())
                                    micro_batch_metrics["rep/full_projector_out_dim"] = float(w.shape[0])
                                    micro_batch_metrics["rep/full_projector_in_dim"] = float(w.shape[1])
                                micro_batch_metrics["rep/use_low_rank_projector"] = 0.0
                                micro_batch_metrics["rep/teacher_pt_initialized"] = 0.0
                                micro_batch_metrics["rep/projector_loaded_from_checkpoint"] = 0.0

                            rep_loss = multi_layer_normalized_mse_loss(
                                student_repr,
                                teacher_repr,
                                position_mask=position_mask,
                                loss_agg_mode=loss_agg_mode,
                                num_layers=num_rep_layers,
                            )
                            micro_batch_metrics.update(
                                compute_rep_alignment_metrics(
                                    student_repr,
                                    teacher_repr,
                                    position_mask=position_mask,
                                    num_layers=num_rep_layers,
                                )
                            )
                            # Backward-compatible aliases used in earlier logs.
                            micro_batch_metrics["rep/norm_mse"] = micro_batch_metrics["rep/subspace_mse"]
                            micro_batch_metrics["rep/cosine_similarity"] = micro_batch_metrics["rep/subspace_cosine"]
                            micro_batch_metrics["actor/rep_loss"] = rep_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/rep_weighted_loss"] = (
                                rep_loss.detach().item() * rep_distillation_effective_coef * loss_scale_factor
                            )
                            micro_batch_metrics["actor/rep_distillation_coef"] = rep_distillation_effective_coef
                            micro_batch_metrics["actor/rep_distillation_crossfade_steps"] = rep_distillation_crossfade_steps
                            rep_term = rep_loss * rep_distillation_effective_coef
                            if policy_loss is None:
                                policy_loss = rep_term
                            else:
                                policy_loss = policy_loss + rep_term

                        if use_jlens_distillation:
                            student_token_ids = None
                            artifacts = None
                            if jalign_uses_fitted_lens(jalign_student_transport):
                                artifacts = getattr(self, "_jalign_student_artifacts", None)
                                if artifacts is None:
                                    mapping = (
                                        None
                                        if jalign_loss_type in {"mse", "opd", "flow_opd"}
                                        else self.config.get("jalign_vocab_mapping")
                                    )
                                    artifacts = JLensArtifacts.load(
                                        self.config.get("student_jlens_checkpoint"),
                                        mapping,
                                        expected_layers=jalign_student_layers,
                                    )
                                    self._jalign_student_artifacts = artifacts
                                student_token_ids = artifacts.token_ids
                            elif jalign_loss_type == "kl":
                                if not hasattr(self, "_jalign_stoch_token_ids"):
                                    self._jalign_stoch_token_ids = load_selected_vocab_ids(
                                        self.config.get("jalign_vocab_mapping")
                                    )
                                student_token_ids = self._jalign_stoch_token_ids

                            student_all_layers = forward_response_hidden_repr(
                                self.actor_module,
                                model_inputs["input_ids"],
                                model_inputs["attention_mask"],
                                model_inputs["position_ids"],
                                model_inputs["response_mask"],
                                use_remove_padding=self.use_remove_padding,
                                use_ulysses_sp=self.use_ulysses_sp,
                                ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
                                positions=jalign_positions,
                                layers="all",
                                last_k=jalign_last_k,
                                first_k=jalign_first_k,
                            )
                            selected_hidden = select_stacked_block_hidden_states(
                                student_all_layers,
                                jalign_student_layers,
                            )
                            adapter_transport_loss = None
                            uses_low_rank_adapter = jalign_student_transport in {"conditional_residual", "identity_adapter"}
                            uses_identity_credit = jalign_gradient_routing == "identity_credit"
                            if uses_low_rank_adapter:
                                adapter_lens = self._get_or_create_low_rank_jlens(
                                    artifacts, identity_adapter=jalign_student_transport == "identity_adapter"
                                )
                                # The legacy conditional mode isolates adapter updates to the
                                # transport MSE.  identity_adapter also receives JOPD gradients.
                                student_transported = adapter_lens(
                                    selected_hidden, detach_params=jalign_student_transport == "conditional_residual"
                                )
                                adapter_prediction = adapter_lens(selected_hidden.detach())
                                target_final = student_all_layers[:, -1].detach()
                                adapter_transport_loss = (adapter_prediction - target_final.unsqueeze(1)).square().mean(dim=-1)
                            if jalign_positions in ("first_k", "last_k"):
                                jalign_mask = build_compact_rep_distillation_position_mask(
                                    model_inputs["response_mask"],
                                    jalign_positions,
                                    jalign_last_k,
                                    jalign_first_k,
                                )
                            elif jalign_positions == "all":
                                jalign_mask = model_inputs["response_mask"]
                            else:
                                raise ValueError("JAlign currently supports all, first_k, or last_k positions")

                            if jalign_student_transport == "stochastic_low_rank":
                                student_transported = stochastic_jlens_transported_residuals(
                                    self.actor_module,
                                    source_layers=jalign_student_layers,
                                    source_residuals=selected_hidden,
                                    input_ids=model_inputs["input_ids"],
                                    attention_mask=model_inputs["attention_mask"],
                                    position_ids=model_inputs["position_ids"],
                                    response_mask=model_inputs["response_mask"],
                                    positions=jalign_positions,
                                    last_k=jalign_last_k,
                                    first_k=jalign_first_k,
                                    rank=jalign_stoch_rank,
                                    eps=jalign_stoch_eps,
                                    probe=jalign_stoch_probe,
                                    use_remove_padding=self.use_remove_padding,
                                    use_ulysses_sp=self.use_ulysses_sp,
                                )
                            elif uses_identity_credit:
                                student_transported = jlens_identity_credit_residuals(
                                    selected_hidden, artifacts=artifacts
                                )
                            elif not uses_low_rank_adapter:
                                student_transported = None

                            jalign_metrics = {}
                            if jalign_loss_type == "mse":
                                if jalign_student_transport == "stochastic_low_rank":
                                    student_jvectors = student_transported
                                else:
                                    student_jvectors = jlens_transported_residuals(
                                        selected_hidden, artifacts=artifacts
                                    )
                                jalign_loss, jalign_metrics = masked_jalign_mse(
                                    model_inputs["teacher_jvectors"],
                                    student_jvectors,
                                    jalign_mask,
                                )
                            elif jalign_loss_type in {"opd", "flow_opd"}:
                                # J-OPD keeps final OPD untouched.  Its auxiliary either shares
                                # final-policy support or freezes a self top-k support per J-layer.
                                if jalign_opd_support == "final_policy" and "student_top_k_ids" not in model_inputs:
                                    raise ValueError(
                                        "J-OPD requires final-policy student_top_k_ids; "
                                        "set LOG_PROB_TOP_K to the OPRD top-k value."
                                    )
                                if self.config.get("jalign_vocab_mapping", None) not in (None, "null"):
                                    raise ValueError(
                                        "J-OPD on final-policy top-k currently requires a shared tokenizer; "
                                        "vocabulary mapping is not applicable."
                                    )
                                if jalign_opd_support == "self_layerwise":
                                    if "jalign_student_top_k_ids" not in model_inputs:
                                        raise ValueError("self-layerwise JOPD requires frozen jalign_student_top_k_ids")
                                    jalign_topk_ids = model_inputs["jalign_student_top_k_ids"]
                                    if jalign_topk_ids.size(1) != len(jalign_student_layers):
                                        raise ValueError("self-layerwise JOPD layer count does not match configured layers")
                                else:
                                    jalign_topk_ids, _ = compact_response_repr_by_positions(
                                        model_inputs["student_top_k_ids"], model_inputs["response_mask"],
                                        jalign_positions, jalign_last_k, jalign_first_k,
                                    )
                                fsdp_context = (
                                    FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                                    if isinstance(self.actor_module, FSDP)
                                    else nullcontext()
                                )
                                with fsdp_context:
                                    student_final_norm, student_lm_head = resolve_final_norm_and_lm_head(
                                        self.actor_module
                                    )
                                    if jalign_loss_type == "flow_opd":
                                        if jflow_token_offset >= student_transported.size(2):
                                            raise ValueError(
                                                "J-Flow token offset must be smaller than the compact response width"
                                            )
                                        flow_topk_ids = jalign_topk_ids[:, jflow_token_offset:]
                                        student_jflow_current_log_probs = layerwise_log_probs_on_policy_topk(
                                            student_transported[:, :, jflow_token_offset:],
                                            policy_topk_ids=flow_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                        student_jflow_anchor_log_probs = layerwise_log_probs_on_policy_topk(
                                            student_transported[:, :, :-jflow_token_offset],
                                            policy_topk_ids=flow_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                    elif uses_low_rank_adapter or uses_identity_credit:
                                        student_jalign_log_probs = (
                                            layerwise_log_probs_on_per_layer_topk(
                                                student_transported, layerwise_topk_ids=jalign_topk_ids,
                                                final_norm=student_final_norm, lm_head=student_lm_head,
                                                temperature=jalign_temperature,
                                            ) if jalign_opd_support == "self_layerwise" else
                                            layerwise_log_probs_on_policy_topk(
                                                student_transported, policy_topk_ids=jalign_topk_ids,
                                                final_norm=student_final_norm, lm_head=student_lm_head,
                                                temperature=jalign_temperature,
                                            )
                                        )
                                    elif jalign_opd_support == "self_layerwise":
                                        student_jalign_log_probs = jlens_log_probs_on_per_layer_topk(
                                            selected_hidden, artifacts=artifacts,
                                            layerwise_topk_ids=jalign_topk_ids,
                                            final_norm=student_final_norm, lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                    elif jalign_student_transport == "stochastic_low_rank":
                                        student_jalign_log_probs = layerwise_log_probs_on_policy_topk(
                                            student_transported,
                                            policy_topk_ids=jalign_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                    else:
                                        student_jalign_log_probs = jlens_log_probs_on_policy_topk(
                                            selected_hidden,
                                            artifacts=artifacts,
                                            policy_topk_ids=jalign_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                    if jalign_loss_type == "flow_opd":
                                        student_raw_log_probs = layerwise_log_probs_on_policy_topk(
                                            selected_hidden,
                                            policy_topk_ids=jalign_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            temperature=jalign_temperature,
                                        )
                                if jalign_loss_type == "flow_opd":
                                    raw_layer_losses = masked_layerwise_opd_policy_losses_from_log_probs(
                                        model_inputs["teacher_jflow_raw_candidate_log_probs"],
                                        student_raw_log_probs,
                                        jalign_mask,
                                    )
                                    flow_layer_losses = masked_jflow_opd_policy_losses_from_log_probs(
                                        model_inputs["teacher_jalign_candidate_log_probs"],
                                        model_inputs["teacher_jflow_anchor_candidate_log_probs"],
                                        student_jflow_current_log_probs,
                                        student_jflow_anchor_log_probs,
                                        jalign_mask,
                                        token_offset=jflow_token_offset,
                                    )
                                    raw_layerwise_loss = raw_layer_losses.mean()
                                    jalign_loss = flow_layer_losses.mean()
                                    jalign_metrics = {
                                        "jalign/valid_tokens": jalign_mask.sum().detach(),
                                        "jflow/loss": jalign_loss.detach(),
                                        "jflow/raw_layerwise_loss": raw_layerwise_loss.detach(),
                                        "jflow/token_offset": torch.tensor(
                                            float(jflow_token_offset), device=jalign_loss.device
                                        ),
                                    }
                                else:
                                    jalign_loss = masked_layerwise_opd_policy_loss_from_log_probs(
                                        model_inputs["teacher_jalign_candidate_log_probs"],
                                        student_jalign_log_probs,
                                        jalign_mask,
                                    )
                                    jalign_metrics = {
                                        "jalign/valid_tokens": jalign_mask.sum().detach(),
                                        "jalign/opd_loss": jalign_loss.detach(),
                                    }
                            else:
                                # Ablation: streamed full-vocab T^2 KL (not the formal OPD pair).
                                if jalign_student_transport != "stochastic_low_rank":
                                    student_transported = jlens_transported_residuals(
                                        selected_hidden, artifacts=artifacts
                                    )
                                fsdp_context = (
                                    FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                                    if isinstance(self.actor_module, FSDP)
                                    else nullcontext()
                                )
                                with fsdp_context:
                                    student_final_norm, student_lm_head = resolve_final_norm_and_lm_head(
                                        self.actor_module
                                    )
                                    jalign_loss, jalign_metrics = masked_jalign_kl_from_transported_residuals(
                                        model_inputs["teacher_jlogits"],
                                        student_transported,
                                        jalign_mask,
                                        final_norm=student_final_norm,
                                        lm_head=student_lm_head,
                                        token_ids=student_token_ids,
                                        temperature=jalign_temperature,
                                    )
                            micro_batch_metrics.update(
                                {key: float(value.detach().item()) for key, value in jalign_metrics.items()}
                            )
                            micro_batch_metrics["actor/jalign_loss"] = jalign_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/jalign_coef"] = jalign_coef
                            micro_batch_metrics["actor/jalign_loss_type"] = {
                                "kl": 0.0,
                                "mse": 1.0,
                                "opd": 2.0,
                                "flow_opd": 3.0,
                            }.get(jalign_loss_type, -1.0)
                            transport_code = {
                                "frozen": 0.0,
                                "stochastic_low_rank": 1.0,
                                "periodic_refresh": 2.0,
                                "conditional_residual": 3.0,
                                "identity_adapter": 4.0,
                            }.get(jalign_student_transport, -1.0)
                            micro_batch_metrics["actor/jalign_student_transport"] = transport_code
                            micro_batch_metrics["actor/jalign_gradient_routing"] = (
                                1.0 if uses_identity_credit else 0.0
                            )
                            if jalign_student_transport == "stochastic_low_rank":
                                micro_batch_metrics["actor/jalign_stoch_rank"] = float(jalign_stoch_rank)
                            if jalign_student_transport == "periodic_refresh":
                                micro_batch_metrics["actor/jalign_refresh_interval"] = float(
                                    self.config.get("jalign_refresh_interval", 25)
                                )
                            if jalign_loss_type == "flow_opd":
                                if policy_loss is None:
                                    raise ValueError("J-Flow requires the final-policy OPD loss")
                                num_jalign_layers = len(jalign_student_layers)
                                policy_loss = aggregate_final_and_layerwise_opd_loss(
                                    policy_loss,
                                    raw_layerwise_loss,
                                    num_intermediate_layers=num_jalign_layers,
                                    intermediate_layer_coef=jalign_coef,
                                    aggregation="legacy_additive",
                                )
                                policy_loss = policy_loss + jflow_coef * jalign_loss
                                micro_batch_metrics["actor/jflow_coef"] = jflow_coef
                                micro_batch_metrics["actor/jflow_total_layers"] = float(
                                    num_jalign_layers + 1
                                )
                                micro_batch_metrics["actor/jflow_identity_credit"] = float(
                                    uses_identity_credit
                                )
                            elif jalign_loss_type == "opd":
                                # Ordinary final-policy OPD is the J_L=I member of this
                                # layer set.  `jalign_loss` is already mean_l L_l, so
                                # multiply it by L before the shared average: with coef=1,
                                # every final/intermediate layer has exactly weight 1.
                                if policy_loss is None:
                                    raise ValueError("J-OPD requires the final-policy OPD loss")
                                num_jalign_layers = len(jalign_student_layers)
                                policy_loss = aggregate_final_and_layerwise_opd_loss(
                                    policy_loss,
                                    jalign_loss,
                                    num_intermediate_layers=num_jalign_layers,
                                    intermediate_layer_coef=jalign_coef,
                                    aggregation=jalign_opd_aggregation,
                                )
                                if jalign_opd_aggregation == "legacy_additive":
                                    final_layer_weight = 1.0
                                    intermediate_layer_weight = jalign_coef / num_jalign_layers
                                    weighted_jalign_loss = jalign_loss.detach().item() * jalign_coef
                                    aggregation_code = 1.0
                                else:
                                    total_jalign_weight = jalign_coef * num_jalign_layers
                                    final_layer_weight = 1.0 / (1.0 + total_jalign_weight)
                                    intermediate_layer_weight = jalign_coef / (1.0 + total_jalign_weight)
                                    weighted_jalign_loss = (
                                        jalign_loss.detach().item()
                                        * total_jalign_weight
                                        / (1.0 + total_jalign_weight)
                                    )
                                    aggregation_code = 0.0
                                micro_batch_metrics["actor/jalign_weighted_loss"] = (
                                    weighted_jalign_loss * loss_scale_factor
                                )
                                micro_batch_metrics["actor/jalign_total_layers"] = float(num_jalign_layers + 1)
                                micro_batch_metrics["actor/jalign_opd_aggregation"] = aggregation_code
                                micro_batch_metrics["actor/jalign_final_layer_weight"] = final_layer_weight
                                micro_batch_metrics["actor/jalign_intermediate_layer_weight"] = intermediate_layer_weight
                                if adapter_transport_loss is not None:
                                    transport_mask = jalign_mask.unsqueeze(1).to(adapter_transport_loss.dtype)
                                    transport_loss = (adapter_transport_loss * transport_mask).sum() / transport_mask.sum().clamp_min(1)
                                    transport_coef = float(self.config.get(
                                        "jalign_identity_adapter_transport_coef"
                                        if jalign_student_transport == "identity_adapter"
                                        else "jalign_conditional_transport_coef",
                                        1.0,
                                    ))
                                    policy_loss = policy_loss + transport_coef * transport_loss
                                    micro_batch_metrics["jalign/adapter_transport_loss"] = transport_loss.detach().item()
                                    micro_batch_metrics["jalign/adapter_transport_coef"] = transport_coef
                            else:
                                jalign_term = jalign_loss * jalign_coef
                                policy_loss = jalign_term if policy_loss is None else policy_loss + jalign_term
                                micro_batch_metrics["actor/jalign_weighted_loss"] = (
                                    jalign_loss.detach().item() * jalign_coef * loss_scale_factor
                                )

                        if pure_forward_kl:
                            assert policy_loss is None, "Pure forward KL cannot be added to another objective"
                            from verl.utils.pure_forward_kl import (
                                load_teacher_readout, prediction_hidden, pure_forward_kl_loss,
                            )
                            if not hasattr(self, "_pure_fkl_teacher_readout"):
                                self._pure_fkl_teacher_readout = load_teacher_readout(
                                    self.config.pure_forward_kl_teacher_model, get_device_id()
                                )
                                self._pure_fkl_lenses = (None, None)
                                if self.config.pure_forward_kl_readout == "jlens":
                                    self._pure_fkl_lenses = tuple(
                                        JLensArtifacts.load(self.config.get(key), None, expected_layers=[20])
                                        for key in ("student_jlens_checkpoint", "teacher_jlens_checkpoint")
                                    )
                            student_hidden = prediction_hidden(self.actor_module, model_inputs)
                            fsdp_context = (
                                FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                                if isinstance(self.actor_module, FSDP) else nullcontext()
                            )
                            with fsdp_context:
                                snorm, shead = resolve_final_norm_and_lm_head(self.actor_module)
                                tnorm, thead = self._pure_fkl_teacher_readout
                                policy_loss, pure_metrics = pure_forward_kl_loss(
                                    student_hidden, model_inputs["teacher_pure_fkl_hidden"], response_mask,
                                    snorm, shead.weight, tnorm, thead, *self._pure_fkl_lenses,
                                    last_k=2000,
                                    final_only=self.config.pure_forward_kl_readout == "final_only",
                                )
                            micro_batch_metrics.update({key: float(value) for key, value in pure_metrics.items()})

                        if use_direct_softmax_distillation and not pure_forward_kl:
                            token_ids = getattr(self, "_direct_softmax_token_ids", None)
                            if token_ids is None:
                                token_ids = load_selected_vocab_ids(
                                    self.config.get("direct_softmax_vocab_mapping")
                                )
                                self._direct_softmax_token_ids = token_ids
                            student_all_layers = forward_response_hidden_repr(
                                self.actor_module,
                                model_inputs["input_ids"],
                                model_inputs["attention_mask"],
                                model_inputs["position_ids"],
                                model_inputs["response_mask"],
                                use_remove_padding=self.use_remove_padding,
                                use_ulysses_sp=self.use_ulysses_sp,
                                ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
                                positions=direct_softmax_positions,
                                layers="all",
                                last_k=direct_softmax_last_k,
                                first_k=direct_softmax_first_k,
                            )
                            student_direct_residuals = select_stacked_block_hidden_states(
                                student_all_layers,
                                direct_softmax_student_layers,
                            )
                            fsdp_context = (
                                FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                                if isinstance(self.actor_module, FSDP)
                                else nullcontext()
                            )
                            with fsdp_context:
                                student_final_norm, student_lm_head = resolve_final_norm_and_lm_head(self.actor_module)
                                student_direct_logits = direct_selected_logits(
                                    student_direct_residuals,
                                    token_ids=token_ids,
                                    final_norm=student_final_norm,
                                    lm_head=student_lm_head,
                                )
                            if direct_softmax_positions in ("first_k", "last_k"):
                                direct_mask = build_compact_rep_distillation_position_mask(
                                    model_inputs["response_mask"],
                                    direct_softmax_positions,
                                    direct_softmax_last_k,
                                    direct_softmax_first_k,
                                )
                            elif direct_softmax_positions == "all":
                                direct_mask = model_inputs["response_mask"]
                            else:
                                raise ValueError("direct softmax alignment supports all, first_k, or last_k positions")
                            direct_loss, _ = masked_direct_softmax_kl(
                                model_inputs["teacher_direct_logits"],
                                student_direct_logits,
                                direct_mask,
                                temperature=direct_softmax_temperature,
                            )
                            direct_term = direct_loss * direct_softmax_coef
                            policy_loss = direct_term if policy_loss is None else policy_loss + direct_term

                        if use_layerwise_opd_distillation:
                            if layerwise_opd_support == "final_policy" and "student_top_k_ids" not in model_inputs:
                                raise ValueError(
                                    "LayerwiseOPD requires final-policy student_top_k_ids; "
                                    "set LOG_PROB_TOP_K to the OPRD top-k value."
                                )
                            if layerwise_opd_support == "self_student":
                                if "layerwise_student_top_k_ids" not in model_inputs:
                                    raise ValueError("self-support LayerwiseOPD requires frozen layerwise_student_top_k_ids")
                                layerwise_topk_ids = model_inputs["layerwise_student_top_k_ids"]
                                if not pair_mode and layerwise_topk_ids.size(1) != len(layerwise_opd_student_layers):
                                    raise ValueError("self-support LayerwiseOPD layer count does not match configured student layers")
                                if layerwise_sampled_layer_slot is not None:
                                    layerwise_topk_ids = layerwise_topk_ids[
                                        :, layerwise_sampled_layer_slot : layerwise_sampled_layer_slot + 1
                                    ]
                            elif layerwise_opd_support == "final_policy":
                                layerwise_topk_ids, _ = compact_response_repr_by_positions(
                                    model_inputs["student_top_k_ids"],
                                    model_inputs["response_mask"],
                                    layerwise_opd_positions,
                                    layerwise_opd_last_k,
                                    layerwise_opd_first_k,
                                )
                            else:
                                if "teacher_layerwise_top_k_ids" not in model_inputs:
                                    raise ValueError("teacher-support LayerwiseOPD requires teacher_layerwise_top_k_ids")
                                layerwise_topk_ids = model_inputs["teacher_layerwise_top_k_ids"]
                                if layerwise_topk_ids.size(1) != len(layerwise_opd_student_layers):
                                    raise ValueError("teacher-support LayerwiseOPD layer count does not match configured student layers")
                            if not pair_mode and layerwise_topk_ids.size(-1) != layerwise_opd_top_k:
                                raise ValueError(
                                    "LayerwiseOPD candidate support has an unexpected width: "
                                    f"got {layerwise_topk_ids.size(-1)} ids, while "
                                    f"layerwise_opd_top_k={layerwise_opd_top_k}."
                                )
                            if self.config.get("layerwise_opd_vocab_mapping", None) not in (None, "null"):
                                raise ValueError(
                                    "LayerwiseOPD candidate support currently requires a shared tokenizer; "
                                    "vocabulary mapping is not applicable."
                                )
                            final_pre_norm_captures = []
                            final_norm_pre_hook = None
                            sampled_source_captures = []
                            sampled_source_hook = None
                            if layerwise_opd_gradient_routing in {
                                "online_jcr",
                                "online_jcr_norm",
                                "online_jcr_activation",
                            } or (layerwise_opd_detach_lm_head and layerwise_objective == "depth_transition"):
                                if self.use_remove_padding:
                                    raise ValueError(
                                        "Online-JCR currently requires use_remove_padding=False"
                                    )
                                student_final_norm, _ = resolve_final_norm_and_lm_head(
                                    self.actor_module
                                )

                                def capture_final_pre_norm(_module, inputs):
                                    final_pre_norm_captures.append(inputs[0])

                                if layerwise_opd_gradient_routing in {"online_jcr", "online_jcr_norm"} or (
                                    layerwise_opd_detach_lm_head and layerwise_objective == "depth_transition"
                                ):
                                    final_norm_pre_hook = student_final_norm.register_forward_pre_hook(
                                        capture_final_pre_norm
                                    )
                                if layerwise_opd_gradient_routing == "online_jcr_activation":
                                    sampled_layer_id = layerwise_opd_student_layers[online_jcr_layer_slot]
                                    decoder_blocks = resolve_decoder_blocks(self.actor_module)
                                    if sampled_layer_id >= len(decoder_blocks):
                                        raise ValueError(
                                            f"sampled Online-JCR layer {sampled_layer_id} is outside "
                                            f"the actor's {len(decoder_blocks)} decoder blocks"
                                        )

                                    def capture_sampled_source(_module, _inputs, output):
                                        sampled_source_captures.append(
                                            output[0] if isinstance(output, tuple) else output
                                        )

                                    sampled_source_hook = decoder_blocks[sampled_layer_id].register_forward_hook(
                                        capture_sampled_source
                                    )
                            select_student_before_stack = (
                                layerwise_opd_support == "final_policy"
                                and layerwise_opd_gradient_routing == "raw"
                                and not layerwise_opd_sample_one_layer
                            )
                            try:
                                student_all_layers = forward_response_hidden_repr(
                                    self.actor_module,
                                    model_inputs["input_ids"],
                                    model_inputs["attention_mask"],
                                    model_inputs["position_ids"],
                                    model_inputs["response_mask"],
                                    use_remove_padding=self.use_remove_padding,
                                    use_ulysses_sp=self.use_ulysses_sp,
                                    ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
                                    positions=layerwise_opd_positions,
                                    layers=(
                                        "blocks:" + ",".join(map(str, layerwise_opd_student_layers))
                                        if select_student_before_stack else "all"
                                    ),
                                    last_k=layerwise_opd_last_k,
                                    first_k=layerwise_opd_first_k,
                                )
                            finally:
                                if final_norm_pre_hook is not None:
                                    final_norm_pre_hook.remove()
                                if sampled_source_hook is not None:
                                    sampled_source_hook.remove()
                            selected_student_layers = (
                                [layerwise_opd_student_layers[layerwise_sampled_layer_slot]]
                                if layerwise_sampled_layer_slot is not None
                                else layerwise_opd_student_layers
                            )
                            student_layerwise_residuals = (
                                student_all_layers
                                if select_student_before_stack
                                else select_stacked_block_hidden_states(
                                    student_all_layers,
                                    selected_student_layers,
                                )
                            )
                            if layerwise_opd_gradient_routing == "frozen_jlens":
                                student_layerwise_residuals = jacobian_backward_identity_residuals(
                                    student_layerwise_residuals,
                                    artifacts=layerwise_opd_jacobian_artifacts,
                                )
                            elif layerwise_opd_gradient_routing in {"online_jcr", "online_jcr_norm"}:
                                if len(final_pre_norm_captures) != 1:
                                    raise RuntimeError(
                                        "Online-JCR expected exactly one final pre-norm residual capture, "
                                        f"got {len(final_pre_norm_captures)}"
                                    )
                                final_pre_norm_residual = extract_teacher_response_hidden_repr(
                                    final_pre_norm_captures[0],
                                    model_inputs["response_mask"],
                                    layerwise_opd_positions,
                                    layers="last",
                                    last_k=layerwise_opd_last_k,
                                    first_k=layerwise_opd_first_k,
                                    compact=True,
                                )
                                student_layerwise_residuals = sampled_online_jacobian_credit_residuals(
                                    student_layerwise_residuals,
                                    final_pre_norm_residual=final_pre_norm_residual,
                                    sampled_layer_slot=online_jcr_layer_slot,
                                    gradient_scale=(
                                        online_jcr_norm_scale
                                        if online_jcr_norm_scale is not None
                                        else 1.0
                                    ),
                                )
                            if layerwise_opd_readout == "frozen_affine":
                                student_layerwise_residuals = frozen_affine_readout_residuals(
                                    student_layerwise_residuals,
                                    artifacts=layerwise_opd_affine_artifacts,
                                )
                            fsdp_context = (
                                FSDP.summon_full_params(self.actor_module, recurse=False, writeback=False)
                                if isinstance(self.actor_module, FSDP)
                                else nullcontext()
                            )
                            with fsdp_context:
                                student_final_norm, student_lm_head = resolve_final_norm_and_lm_head(self.actor_module)
                                if sampled_adjacent:
                                    from verl.utils.sampled_adjacent import sampled_adjacent_readout
                                    if len(layerwise_opd_student_layers) != student_all_layers.size(1) - 1:
                                        raise ValueError("adjacent source count must equal decoder block count minus one")
                                    current_final = compact_response_repr_by_positions(
                                        log_prob_for_loss, model_inputs["response_mask"],
                                        layerwise_opd_positions, layerwise_opd_last_k, layerwise_opd_first_k)[0]
                                    student_layerwise_log_probs = sampled_adjacent_readout(
                                        student_all_layers, model_inputs["layerwise_opd_sampled_slot"],
                                        current_final, layerwise_topk_ids, final_norm=student_final_norm,
                                        lm_head=student_lm_head, token_chunk_size=layerwise_opd_token_chunk_size,
                                        temperature=layerwise_opd_temperature)
                                elif pair_mode:
                                    from verl.utils.transition_support import path_readout
                                    student_layerwise_log_probs = path_readout(
                                        student_all_layers, layerwise_opd_student_layers, layerwise_topk_ids,
                                        student_final_norm, student_lm_head, layerwise_opd_token_chunk_size)
                                elif layerwise_opd_support in {"self_student", "teacher_layerwise"}:
                                    student_layerwise_log_probs = layerwise_log_probs_on_per_layer_topk(
                                        student_layerwise_residuals,
                                        layerwise_topk_ids=layerwise_topk_ids,
                                        final_norm=student_final_norm,
                                        lm_head=student_lm_head,
                                        token_chunk_size=layerwise_opd_token_chunk_size,
                                        temperature=layerwise_opd_temperature,
                                    )
                                else:
                                    student_layerwise_log_probs = layerwise_log_probs_on_policy_topk(
                                        student_layerwise_residuals,
                                        policy_topk_ids=layerwise_topk_ids,
                                        final_norm=student_final_norm,
                                        lm_head=student_lm_head,
                                        token_chunk_size=layerwise_opd_token_chunk_size,
                                        temperature=layerwise_opd_temperature,
                                        detach_lm_head=bool(
                                            self.config.get("layerwise_opd_detach_lm_head", False)
                                        ),
                                    )
                                student_final_detached_lm_head_log_probs = None
                                if (
                                    layerwise_opd_detach_lm_head
                                    and layerwise_objective == "depth_transition"
                                ):
                                    if len(final_pre_norm_captures) != 1:
                                        raise RuntimeError("DepthTransition expected one final pre-norm capture")
                                    final_residual = extract_teacher_response_hidden_repr(
                                        final_pre_norm_captures[0], model_inputs["response_mask"],
                                        layerwise_opd_positions, layers="last",
                                        last_k=layerwise_opd_last_k,
                                        first_k=layerwise_opd_first_k, compact=True,
                                    )
                                    student_final_detached_lm_head_log_probs = (
                                        layerwise_log_probs_on_policy_topk(
                                            final_residual.unsqueeze(1),
                                            policy_topk_ids=layerwise_topk_ids,
                                            final_norm=student_final_norm,
                                            lm_head=student_lm_head,
                                            token_chunk_size=layerwise_opd_token_chunk_size,
                                            temperature=layerwise_opd_temperature,
                                            detach_lm_head=True,
                                        )
                                    )
                            if layerwise_opd_positions in ("first_k", "last_k"):
                                layerwise_mask = build_compact_rep_distillation_position_mask(
                                    model_inputs["response_mask"],
                                    layerwise_opd_positions,
                                    layerwise_opd_last_k,
                                    layerwise_opd_first_k,
                                )
                            elif layerwise_opd_positions == "all":
                                layerwise_mask = model_inputs["response_mask"]
                            else:
                                raise ValueError("LayerwiseOPD supports all, first_k, or last_k positions")
                            teacher_layerwise_candidate_log_probs = model_inputs[
                                "teacher_layerwise_candidate_log_probs"
                            ]
                            # For the independent-routing ablation, recompute
                            # ExpertAlign separately at each intermediate layer
                            # from that layer's teacher/base/student readouts.
                            if self.config.get("expertalign_layerwise_per_layer", False) and "expertalign_teacher_layerwise_0" in model_inputs:
                                from verl.utils.expertalign_router import expertalign_route_log_probs
                                per_teacher = torch.stack([
                                    model_inputs[f"expertalign_teacher_layerwise_{idx}"].float()
                                    for idx in range(3)
                                ], dim=0)
                                per_base = model_inputs["expertalign_base_layerwise_candidate_log_probs"].float()
                                routed_layers = []
                                routed_masks = []
                                for layer_slot in range(per_teacher.size(2)):
                                    routed, lw, _, _ = expertalign_route_log_probs(
                                        student_layerwise_log_probs[:, layer_slot],
                                        per_teacher[:, :, layer_slot],
                                        per_base[:, layer_slot],
                                        weighting="cosine",
                                        alignment_epsilon=1e-6,
                                    )
                                    routed_layers.append(routed)
                                    routed_masks.append(lw.sum(0) > 0)
                                teacher_layerwise_candidate_log_probs = torch.stack(routed_layers, dim=1).to(
                                    teacher_layerwise_candidate_log_probs.dtype
                                )
                                # The loss API takes one response-token mask
                                # shared by all layer slots.  Routing weights
                                # are layer-specific, so reduce their masks
                                # across layers before applying the ordinary
                                # layerwise objective.
                                layerwise_mask = torch.stack(routed_masks, dim=1).any(dim=1)
                            if "expertalign_layerwise_mask" in model_inputs:
                                layerwise_mask = layerwise_mask * model_inputs["expertalign_layerwise_mask"]
                            micro_batch_metrics["actor/layerwise_opd_objective"] = (
                                1.0 if layerwise_objective in {"depth_transition", "sampled_adjacent_transition"} else 0.0
                            )
                            expected_teacher_layers = (
                                1
                                if layerwise_sampled_layer_slot is not None
                                else len(layerwise_opd_student_layers)
                            )
                            if sampled_adjacent:
                                expected_teacher_layers = 2
                            if pair_mode:
                                expected_teacher_layers = 4
                            if teacher_layerwise_candidate_log_probs.size(1) != expected_teacher_layers:
                                raise ValueError(
                                    "teacher LayerwiseOPD layer count does not match the actor readout: "
                                    f"got {teacher_layerwise_candidate_log_probs.size(1)}, "
                                    f"expected {expected_teacher_layers}"
                                )
                            if sampled_adjacent:
                                per_layer_losses = masked_layerwise_transition_opd_policy_losses_from_log_probs(
                                    teacher_layerwise_candidate_log_probs[:, :1], student_layerwise_log_probs[:, :1],
                                    teacher_layerwise_candidate_log_probs[:, 1], student_layerwise_log_probs[:, 1],
                                    layerwise_mask)
                                micro_batch_metrics["actor/adjacent_pair_index"] = float(layerwise_sampled_layer_slot)
                            elif pair_mode:
                                from verl.utils.transition_support import select_indices, losses
                                selected_indices = select_indices(
                                    layerwise_topk_ids, model_inputs["transition_snapshot_log_probs"],
                                    teacher_layerwise_candidate_log_probs, pair_selection)
                                per_layer_losses = losses(student_layerwise_log_probs,
                                    teacher_layerwise_candidate_log_probs, selected_indices, layerwise_mask)
                            elif layerwise_objective == "depth_transition":
                                if layerwise_opd_support != "final_policy":
                                    raise ValueError(
                                        "depth_transition LayerwiseOPD requires final_policy support"
                                    )
                                teacher_final = compact_response_repr_by_positions(
                                    model_inputs["teacher_on_student_log_probs"],
                                    model_inputs["response_mask"],
                                    layerwise_opd_positions,
                                    layerwise_opd_last_k,
                                    layerwise_opd_first_k,
                                )[0]
                                if student_final_detached_lm_head_log_probs is not None:
                                    student_final = student_final_detached_lm_head_log_probs[:, 0]
                                else:
                                    student_final = compact_response_repr_by_positions(
                                        log_prob_for_loss,
                                        model_inputs["response_mask"],
                                        layerwise_opd_positions,
                                        layerwise_opd_last_k,
                                        layerwise_opd_first_k,
                                    )[0]
                                if teacher_layerwise_candidate_log_probs.size(1) != len(layerwise_opd_student_layers):
                                    raise ValueError(
                                        "depth_transition requires all configured intermediate layer targets"
                                    )
                                per_layer_losses = masked_layerwise_transition_opd_policy_losses_from_log_probs(
                                    teacher_layerwise_candidate_log_probs,
                                    student_layerwise_log_probs,
                                    teacher_final,
                                    student_final,
                                    layerwise_mask,
                                    delta_align=delta_align,
                                    reverse_transition_kl=reverse_transition_kl,
                                    diagnostics=micro_batch_metrics,
                                    weight_only=weight_only,
                                    center_only=center_only,
                                )
                                micro_batch_metrics["actor/depth_delta_align_enabled"] = float(delta_align)
                                micro_batch_metrics["actor/layerwise_opd_detach_lm_head"] = float(
                                    bool(self.config.get("layerwise_opd_detach_lm_head", False))
                                )
                            elif layerwise_objective == "absolute":
                                per_layer_losses = masked_layerwise_opd_policy_losses_from_log_probs(
                                    teacher_layerwise_candidate_log_probs,
                                    student_layerwise_log_probs,
                                    layerwise_mask,
                                    domain_is_code=(
                                        domain_is_code_from_sources(
                                            model_inputs.get("data_source"),
                                            batch_size=layerwise_mask.shape[0],
                                            device=layerwise_mask.device,
                                        )
                                        if loss_agg_mode == "domain-equal-token-mean"
                                        else None
                                    ),
                                )
                            else:
                                raise ValueError(
                                    f"unknown layerwise_opd_objective={layerwise_objective!r}; "
                                    "expected 'absolute' or 'depth_transition'"
                                )
                            adaptive_weights = None
                            if bool(self.config.get("layerwise_opd_adaptive", False)):
                                beta = float(self.config.get("layerwise_opd_adaptive_ema_beta", 0.9))
                                temperature = float(self.config.get("layerwise_opd_adaptive_temperature", 1.0))
                                if not 0.0 <= beta < 1.0:
                                    raise ValueError("layerwise_opd_adaptive_ema_beta must be in [0, 1)")
                                if temperature <= 0:
                                    raise ValueError("layerwise_opd_adaptive_temperature must be positive")
                                current = per_layer_losses.detach().float()
                                if self.layerwise_adaptive_ema is None or self.layerwise_adaptive_ema.numel() != current.numel():
                                    self.layerwise_adaptive_ema = current.clone()
                                    self.layerwise_adaptive_prev_ema = current.clone()
                                else:
                                    self.layerwise_adaptive_prev_ema.copy_(self.layerwise_adaptive_ema)
                                    self.layerwise_adaptive_ema.mul_(beta).add_(current, alpha=1.0 - beta)
                                improvement = self.layerwise_adaptive_prev_ema - self.layerwise_adaptive_ema
                                centered = improvement - improvement.mean()
                                adaptive_weights = torch.softmax(centered / temperature, dim=0).to(per_layer_losses.dtype)
                                micro_batch_metrics["actor/layerwise_opd_adaptive_weight_mean"] = float(adaptive_weights.mean().item())
                                for slot, weight in enumerate(adaptive_weights):
                                    micro_batch_metrics[f"actor/layerwise_opd_adaptive_weight_{slot}"] = float(weight.item())
                            if transition_weights is not None:
                                if per_layer_losses.numel() != len(transition_weights):
                                    raise ValueError("transition weight count does not match transition loss count")
                                fixed_weights_tensor = per_layer_losses.new_tensor(transition_weights)
                                raw_layerwise_loss = (per_layer_losses * fixed_weights_tensor).sum()
                                for slot, weight in enumerate(transition_weights):
                                    micro_batch_metrics[f"actor/layerwise_opd_transition_weight_{slot}"] = weight
                            else:
                                raw_layerwise_loss = per_layer_losses.mean()
                            if layerwise_opd_diagnostics:
                                diagnostic_layers = (
                                    [layerwise_opd_student_layers[layerwise_sampled_layer_slot]]
                                    if layerwise_sampled_layer_slot is not None
                                    else layerwise_opd_student_layers
                                )
                                distribution_diagnostics = layerwise_opd_candidate_distribution_diagnostics(
                                    teacher_layerwise_candidate_log_probs,
                                    student_layerwise_log_probs,
                                    layerwise_mask,
                                )
                                final_logit_cotangent = torch.autograd.grad(
                                    pg_loss,
                                    log_prob_for_loss,
                                    retain_graph=True,
                                    create_graph=False,
                                )[0].detach()
                                final_cotangent_norm = torch.linalg.vector_norm(final_logit_cotangent.float())
                                micro_batch_metrics["actor/diagnostics_final_opd_loss"] = (
                                    pg_loss.detach().item() * loss_scale_factor
                                )
                                micro_batch_metrics["actor/diagnostics_final_logit_cotangent_norm"] = float(
                                    final_cotangent_norm.item()
                                )
                                for layer_slot, layer_id in enumerate(diagnostic_layers):
                                    layer_cotangent = torch.autograd.grad(
                                        per_layer_losses[layer_slot],
                                        student_layerwise_log_probs,
                                        retain_graph=True,
                                        create_graph=False,
                                    )[0][:, layer_slot].detach()
                                    layer_cotangent_norm = torch.linalg.vector_norm(layer_cotangent.float())
                                    flat_final = final_logit_cotangent.float().reshape(-1)
                                    flat_layer = layer_cotangent.float().reshape(-1)
                                    cosine_denominator = final_cotangent_norm * layer_cotangent_norm
                                    cosine = torch.where(
                                        cosine_denominator > 0,
                                        torch.dot(flat_final, flat_layer) / cosine_denominator,
                                        torch.zeros_like(cosine_denominator),
                                    )
                                    prefix = f"actor/diagnostics_layer_{layer_id}"
                                    micro_batch_metrics[f"{prefix}_opd_loss"] = (
                                        per_layer_losses[layer_slot].detach().item() * loss_scale_factor
                                    )
                                    micro_batch_metrics[f"{prefix}_logit_cotangent_norm"] = float(
                                        layer_cotangent_norm.item()
                                    )
                                    micro_batch_metrics[f"{prefix}_final_logit_cotangent_cosine"] = float(
                                        cosine.item()
                                    )
                                    for metric_name, values in distribution_diagnostics.items():
                                        micro_batch_metrics[f"{prefix}_{metric_name}"] = float(
                                            values[layer_slot].item()
                                        )
                            if layerwise_opd_gradient_routing == "online_jcr_activation":
                                if len(sampled_source_captures) != 1:
                                    raise RuntimeError(
                                        "Activation-only Online-JCR expected exactly one sampled source "
                                        f"capture, got {len(sampled_source_captures)}"
                                    )
                                layerwise_loss = activation_only_online_jcr_layerwise_loss(
                                    per_layer_losses,
                                    sampled_source_residual=sampled_source_captures[0],
                                    routed_cotangent=online_jcr_routed_cotangent,
                                    sampled_layer_slot=online_jcr_layer_slot,
                                )
                            else:
                                layerwise_loss = (
                                    (per_layer_losses * adaptive_weights.detach()).sum()
                                    if adaptive_weights is not None
                                    else raw_layerwise_loss
                                )
                            if policy_loss is None:
                                raise ValueError("LayerwiseOPD requires the final-policy OPD loss")
                            num_layerwise_layers = (
                                1
                                if layerwise_sampled_layer_slot is not None
                                else len(layerwise_opd_student_layers)
                            )
                            policy_loss = aggregate_final_and_layerwise_opd_loss(
                                policy_loss,
                                layerwise_loss,
                                num_intermediate_layers=num_layerwise_layers,
                                intermediate_layer_coef=layerwise_opd_coef,
                                aggregation=layerwise_opd_aggregation,
                            )
                            if layerwise_opd_aggregation == "legacy_additive":
                                final_layer_weight = 1.0
                                intermediate_layer_weight = layerwise_opd_coef / num_layerwise_layers
                                aggregation_code = 1.0
                            else:
                                total_layerwise_weight = layerwise_opd_coef * num_layerwise_layers
                                final_layer_weight = 1.0 / (1.0 + total_layerwise_weight)
                                intermediate_layer_weight = layerwise_opd_coef / (1.0 + total_layerwise_weight)
                                aggregation_code = 0.0
                            micro_batch_metrics["actor/layerwise_opd_loss"] = (
                                layerwise_loss.detach().item() * loss_scale_factor
                            )
                            micro_batch_metrics["actor/layerwise_opd_total_layers"] = float(num_layerwise_layers + 1)
                            micro_batch_metrics["actor/layerwise_opd_aggregation"] = aggregation_code
                            micro_batch_metrics["actor/layerwise_opd_final_layer_weight"] = final_layer_weight
                            micro_batch_metrics["actor/layerwise_opd_intermediate_layer_weight"] = (
                                intermediate_layer_weight
                            )
                            routing_code = {
                                "raw": 0.0,
                                "frozen_jlens": 1.0,
                                "online_jcr": 2.0,
                                "online_jcr_activation": 3.0,
                                "online_jcr_norm": 4.0,
                            }[
                                layerwise_opd_gradient_routing
                            ]
                            micro_batch_metrics["actor/layerwise_opd_gradient_routing"] = routing_code
                            micro_batch_metrics["actor/layerwise_opd_readout"] = (
                                1.0 if layerwise_opd_readout == "frozen_affine" else 0.0
                            )
                            micro_batch_metrics["actor/layerwise_opd_jacobian_backward"] = float(
                                layerwise_opd_gradient_routing == "frozen_jlens"
                            )
                            micro_batch_metrics["actor/layerwise_sample_one_layer"] = float(
                                layerwise_opd_sample_one_layer
                            )
                            if layerwise_sampled_layer_slot is not None:
                                micro_batch_metrics["actor/layerwise_sampled_layer"] = float(
                                    layerwise_opd_student_layers[layerwise_sampled_layer_slot]
                                )
                            if online_jcr_layer_slot is not None:
                                micro_batch_metrics["actor/online_jcr_sampled_layer"] = float(
                                    layerwise_opd_student_layers[online_jcr_layer_slot]
                                )
                            if online_jcr_norm_scale is not None:
                                micro_batch_metrics["actor/online_jcr_raw_cotangent_norm"] = float(
                                    online_jcr_raw_cotangent_norm.detach().item()
                                )
                                micro_batch_metrics["actor/online_jcr_routed_cotangent_norm"] = float(
                                    online_jcr_routed_cotangent_norm.detach().item()
                                )
                                micro_batch_metrics["actor/online_jcr_norm_scale"] = float(
                                    online_jcr_norm_scale.detach().item()
                                )

                        if use_att_distillation:
                            att_position_mask = None
                            if att_distillation_positions == "first_k":
                                att_position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            elif att_distillation_positions == "last_k":
                                att_position_mask = build_compact_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            elif att_distillation_positions == "all":
                                att_position_mask = build_rep_distillation_position_mask(
                                    response_mask,
                                    att_distillation_positions,
                                    att_distillation_last_k,
                                    att_distillation_first_k,
                                )
                            original_seqlen, att_context_start, att_context_len = get_att_distillation_context_metadata(
                                model_inputs["input_ids"],
                                model_inputs["attention_mask"],
                                model_inputs["position_ids"],
                                response_mask,
                                positions=att_distillation_positions,
                                first_k=att_distillation_first_k,
                                last_k=att_distillation_last_k,
                                max_context_len=int(self.config.get("att_distillation_max_key_len", 4096)),
                            )
                            teacher_attn = model_inputs["teacher_attn_rows"]
                            num_att_layers = student_attn.size(1) if multi_layer_att else None
                            # Run att forward before backward, but merge into one backward pass.
                            # FSDP cannot safely run two sequential backward() calls on the same module.
                            student_attn = self._forward_response_attn_rows(model_inputs)
                            att_loss = attention_distillation_loss(
                                student_attn,
                                teacher_attn,
                                loss_type=att_distillation_loss_type,
                                position_mask=att_position_mask,
                                loss_agg_mode=loss_agg_mode,
                                temperature=att_distillation_temperature,
                                num_layers=num_att_layers,
                            )
                            micro_batch_metrics["att/valid_query_fraction"] = att_distillation_query_valid_fraction(
                                response_mask,
                                att_distillation_positions,
                                att_context_start,
                                att_context_len,
                                last_k=att_distillation_last_k,
                                first_k=att_distillation_first_k,
                                original_seqlen=original_seqlen,
                            )
                            micro_batch_metrics["att/context_len"] = float(att_context_len)
                            micro_batch_metrics["att/key_len_student"] = float(student_attn.shape[-1])
                            micro_batch_metrics["att/loss"] = att_loss.detach().item()
                            if num_att_layers is not None:
                                micro_batch_metrics["att/num_layers"] = num_att_layers
                            micro_batch_metrics["actor/att_loss"] = att_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/att_weighted_loss"] = (
                                att_loss.detach().item() * att_distillation_coef * loss_scale_factor
                            )
                            micro_batch_metrics["actor/att_distillation_coef"] = att_distillation_coef
                            del student_attn

                    if policy_loss is None and att_loss is None:
                        raise ValueError(
                            "No loss computed: enable use_rep_distillation, use_att_distillation, or policy gradient training."
                        )

                    total_loss = policy_loss
                    if att_loss is not None:
                        att_term = att_loss * att_distillation_coef
                        total_loss = att_term if total_loss is None else total_loss + att_term

                    (total_loss * loss_scale_factor).backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, update_diagnostics = self._optimizer_step(
                    diagnose_update_norm=layerwise_opd_diagnostics
                )
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if update_diagnostics is not None:
                    mini_batch_metrics.update(update_diagnostics)
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
