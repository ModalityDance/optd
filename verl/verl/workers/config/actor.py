# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "VALID_LOSS_AGG_MODES",
    "PolicyLossConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]

# Hydra/FSDPActorConfig and agg_loss must stay in lockstep. Adding a mode here
# is not enough: agg_loss and every call that forwards actor.loss_agg_mode
# must accept it (domain-equal also needs per-row data_source).
VALID_LOSS_AGG_MODES = (
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
    "domain-equal-token-mean",
)


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. See ``VALID_LOSS_AGG_MODES``.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        use_rep_distillation (bool): Whether to add representation-level distillation loss.
        rep_distillation_coef (float): Coefficient for representation distillation MSE loss.
        rep_distillation_only (bool): If True, optimize only representation MSE (skip policy gradient loss).
        rep_distillation_positions (str): ``last``, ``all``, ``last_k``, or ``first_k`` response tokens for rep loss.
        rep_distillation_last_k (int): Number of trailing response tokens when ``rep_distillation_positions=last_k``.
        rep_distillation_first_k (int): Number of leading response tokens when ``rep_distillation_positions=first_k``.
        rep_distillation_layers (str): ``last``, ``all``, ``even``, or ``odd`` transformer layers for rep loss.
        rep_projector_mode (str): Cross-arch rep bridge: ``full`` (Linear d_S->d_T) or ``low_rank`` (P_S/P_T).
        rep_low_rank (int): Subspace rank r when ``rep_projector_mode=low_rank``.
        rep_low_rank_init_checkpoint (Optional[str]): Path to pre-experiment 2 ``ps_bank.pt`` or ``direct_bank.pt``.
        rep_ps_projector (str): Student P_S bridge in low-rank modes: ``linear``, ``mlp``, or ``auto`` (from checkpoint).
        rep_mlp_hidden_mult (int): MLP hidden width multiplier when ``rep_ps_projector=mlp`` (hidden=max(r, mult*r)).
        rep_freeze_ps (bool): If True, freeze low-rank student projectors P_S (only student LLM is trained).
        rep_head_rank (int): Head subspace rank for ``low_rank_residual`` (frozen top PCA bridge).
        rep_head_init_checkpoint (Optional[str]): Head ``ps_bank.pt`` for residual mode (e.g. rank_16).
        rep_freeze_ps (bool): If True, keep low-rank student projectors P_S frozen (only update student LLM).
        use_att_distillation (bool): Whether to add attention distillation loss.
        att_distillation_coef (float): Coefficient for attention distillation loss.
        att_distillation_layers (str): Layer selection for attention distillation.
        att_distillation_positions (str): Response query positions for attention distillation.
        att_distillation_last_k (int): Trailing response tokens when ``att_distillation_positions=last_k``.
        att_distillation_first_k (int): Leading response tokens when ``att_distillation_positions=first_k``.
        att_distillation_max_key_len (int): Truncated context length for memory-efficient attention extraction.
        att_distillation_loss (str): ``kl`` or ``mse`` over attention key distributions.
        att_distillation_temperature (float): Softmax temperature for KL attention distillation.
        use_jlens_distillation (bool): Enable JLens-based intermediate alignment.
        jalign_loss_type (str): ``opd`` (J-OPD KL-reward score-function), ``kl`` (T^2 KL ablation),
            or ``mse`` (L2-normalized J-vectors).
        jalign_student_transport (str): ``frozen``, ``periodic_refresh``, or ``stochastic_low_rank``.
        jalign_gradient_routing (str): ``jacobian`` differentiates through ``J h``;
            ``identity_credit`` keeps the ``J h`` forward target but uses an identity
            surrogate Jacobian for credit assignment.
        jalign_opd_aggregation (str): ``equal_weight`` or historical
            ``legacy_additive`` final-plus-mean-intermediate weighting.
        jalign_stoch_rank (int): Hutchinson probe count for stochastic transport.
        jalign_stoch_eps (float): Finite-difference step for stochastic JVPs.
        jalign_stoch_probe (str): ``rademacher`` or ``gaussian``.
        jalign_refresh_interval (int): Re-fit student lens every N steps (``periodic_refresh``).
        jalign_refresh_probe_data (Optional[str]): Fixed probe corpus for refresh fitting.
        jalign_refresh_text_key (str): Text field in the probe corpus.
        jalign_refresh_num_prompts (int): Prompt count for each refresh fit.
        jalign_refresh_dim_batch (int): Output-dimension batch size for fitting.
        jalign_refresh_max_seq_len (int): Max probe sequence length.
        jalign_refresh_skip_first (int): Skip early positions when fitting.
        jalign_refresh_seed (int): Deterministic probe selection seed.
        jalign_refresh_cuda_visible_devices (Optional[str]): GPU id(s) for refresh fitting; may share a training GPU when headroom permits.
        jalign_coef (float): For J-OPD, relative weight of each intermediate J-layer
            versus the final OPD layer (1.0 gives equal per-layer weight); for KL/MSE,
            weight of the JAlign auxiliary term.
        jalign_temperature (float): Softmax temperature for ``opd`` / ``kl`` readouts.
        jalign_positions (str): Response positions for JAlign (``all``, ``last_k``, or ``first_k``).
        jalign_student_layers (str): Comma-separated post-block source layer indices.
        jalign_teacher_layers (str): Comma-separated post-block source layer indices.
        student_jlens_checkpoint (Optional[str]): Student ``lens.pt`` or containing directory (frozen mode).
        teacher_jlens_checkpoint (Optional[str]): Teacher ``lens.pt`` or containing directory.
        jalign_vocab_mapping (Optional[str]): Vocabulary subset JSON (``kl`` ablation only; ``opd`` requires null).
        use_direct_softmax_distillation: Enable the no-JLens direct-logit KL control.
        direct_softmax_*: Coefficient, temperature, response positions, layers, and shared vocabulary for that control.
        layerwise_opd_coef: Relative weight of each intermediate raw-OPD layer versus
            final OPD (1.0 gives equal per-layer weight).
        layerwise_opd_aggregation: ``equal_weight`` treats final OPD and every
            intermediate layer as separate terms; ``legacy_additive`` restores
            ``L_final + coef * mean_l(L_l)`` from the early LayerwiseOPD+OPD run.
        layerwise_opd_sample_one_layer: Sample exactly one configured intermediate
            raw LayerwiseOPD layer per PPO minibatch for both teacher and actor
            readouts, as an unbiased estimator of the full intermediate-layer
            mean. Requires raw routing and ``legacy_additive`` aggregation.
        layerwise_opd_jacobian_backward: Keep raw LayerwiseOPD readouts in the
            forward pass and route intermediate gradients through the frozen
            student JLens Jacobian. Legacy compatibility flag; new launches
            should use ``layerwise_opd_gradient_routing=frozen_jlens``.
        layerwise_opd_gradient_routing: ``raw``, ``frozen_jlens``, or
            ``online_jcr``. Online-JCR uses the current input's exact actor
            suffix Jacobian in backward and requires no fitted lens.
        layerwise_opd_readout: ``raw`` or ``frozen_affine``. The latter applies
            frozen, layer-specific affine translators before final norm/LM head.
        layerwise_opd_teacher_target: ``layerwise`` reads the configured teacher
            layers. ``final_prediction`` broadcasts the teacher final-policy
            log-probabilities to every student intermediate readout.
        layerwise_opd_diagnostics: Diagnostic-only logging of per-layer losses,
            candidate-distribution statistics, logit cotangent norms/cosines,
            and the actual optimizer update norm. Disabled by default.
        layerwise_opd_diagnostics_interval: Run the heavier diagnostic block
            every N global training steps when diagnostics are enabled.
        layerwise_opd_online_jcr_norm_chunk_max_tokens: Maximum effective tokens
            per internal Online-JCR norm VJP chunk. Zero preserves the unchunked
            legacy prepass; chunking does not change the outer shared norm scale.
        layerwise_opd_token_chunk_size: Token width of the streamed full-vocabulary
            Norm+LM-head projection. Smaller chunks reduce peak activation memory
            without changing selected log-probs or gradients.
        layerwise_opd_park_teacher_repr_on_cpu: ``auto`` parks teacher hidden
            states on CPU only when there are at least eight layerwise layers
            (AllBlocks). ``False`` keeps them on GPU (7/13/19); ``True`` forces
            the AllBlocks park. This does not change the OPD objective.
        layerwise_opd_student_jlens_checkpoint: Student JLens used only by the
            Jacobian-backward LayerwiseOPD mode.
        layerwise_opd_support: ``final_policy`` shares final OPRD's top-k ids;
            ``self_student`` freezes a separate top-k support for each intermediate student layer.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        # Hot-swapped by periodic JLens refresh (idea.md §1).
        "student_jlens_checkpoint",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    use_rep_distillation: bool = False
    rep_distillation_coef: float = 1.0
    rep_distillation_crossfade_steps: int = 0
    rep_distillation_only: bool = False
    rep_distillation_positions: str = "last"
    rep_distillation_last_k: int = 32
    rep_distillation_first_k: int = 50
    rep_distillation_layers: str = "last"
    rep_projector_mode: str = "full"
    rep_low_rank: int = 256
    rep_low_rank_init_checkpoint: Optional[str] = None
    rep_ps_projector: str = "auto"
    rep_mlp_hidden_mult: int = 4
    rep_freeze_ps: bool = False
    rep_head_rank: int = 16
    rep_head_init_checkpoint: Optional[str] = None
    use_att_distillation: bool = False
    att_distillation_coef: float = 1.0
    att_distillation_layers: str = "last"
    att_distillation_positions: str = "last"
    att_distillation_last_k: int = 32
    att_distillation_first_k: int = 50
    att_distillation_max_key_len: int = 4096
    att_distillation_loss: str = "kl"
    att_distillation_temperature: float = 1.0
    use_jlens_distillation: bool = False
    jalign_loss_type: str = "kl"
    jalign_opd_support: str = "final_policy"
    jalign_student_transport: str = "frozen"
    jalign_gradient_routing: str = "jacobian"
    jalign_opd_aggregation: str = "equal_weight"
    jflow_coef: float = 1.0
    jflow_token_offset: int = 8
    jalign_stoch_rank: int = 8
    jalign_stoch_eps: float = 1.0e-3
    jalign_stoch_probe: str = "rademacher"
    jalign_conditional_rank: int = 1
    jalign_conditional_lr: float = 2.0e-3
    jalign_conditional_transport_coef: float = 1.0
    jalign_identity_adapter_rank: int = 8
    jalign_identity_adapter_lr: float = 2.0e-3
    jalign_identity_adapter_transport_coef: float = 1.0
    jalign_refresh_interval: int = 25
    jalign_refresh_probe_data: Optional[str] = None
    jalign_refresh_response_token_data: Optional[str] = None
    jalign_refresh_text_key: str = "prompt"
    jalign_refresh_num_prompts: int = 100
    jalign_refresh_dim_batch: int = 64
    jalign_refresh_max_seq_len: int = 128
    jalign_refresh_max_response_positions: int = 2000
    jalign_refresh_skip_first: int = 16
    jalign_refresh_seed: int = 42
    jalign_refresh_cuda_visible_devices: Optional[str] = None
    jalign_coef: float = 1.0
    jalign_temperature: float = 1.0
    jalign_positions: str = "last_k"
    jalign_last_k: int = 64
    jalign_first_k: int = 64
    jalign_student_layers: str = "8,14,20"
    jalign_teacher_layers: str = "8,14,20"
    student_jlens_checkpoint: Optional[str] = None
    teacher_jlens_checkpoint: Optional[str] = None
    primary_teacher_jlens_checkpoint: Optional[str] = None
    secondary_teacher_jlens_checkpoint: Optional[str] = None
    jalign_vocab_mapping: Optional[str] = None
    use_direct_softmax_distillation: bool = False
    pure_forward_kl: bool = False
    pure_forward_kl_readout: str = "raw"
    pure_forward_kl_teacher_model: Optional[str] = None
    direct_softmax_coef: float = 1.0
    direct_softmax_temperature: float = 1.0
    direct_softmax_positions: str = "last_k"
    direct_softmax_last_k: int = 64
    direct_softmax_first_k: int = 64
    direct_softmax_student_layers: str = "8,14,20"
    direct_softmax_teacher_layers: str = "8,14,20"
    direct_softmax_vocab_mapping: Optional[str] = None
    use_layerwise_opd_distillation: bool = False
    layerwise_opd_coef: float = 1.0
    expertalign_layerwise_per_layer: bool = False
    layerwise_opd_adaptive: bool = False
    layerwise_opd_adaptive_ema_beta: float = 0.9
    layerwise_opd_adaptive_temperature: float = 1.0
    layerwise_opd_objective: str = "absolute"
    layerwise_opd_transition_weights: str = ""
    layerwise_opd_delta_align: bool = False
    layerwise_opd_reverse_transition_kl: bool = False
    layerwise_opd_weight_only: bool = False
    layerwise_opd_center_only: bool = False
    layerwise_opd_pair_selection: str = "none"
    trim_common_padding: bool = False
    layerwise_opd_pair_selection: str = "none"
    trim_common_padding: bool = False
    layerwise_opd_aggregation: str = "equal_weight"
    layerwise_opd_sample_one_layer: bool = False
    layerwise_opd_temperature: float = 1.0
    layerwise_opd_top_k: int = 16
    layerwise_opd_support: str = "final_policy"
    layerwise_opd_positions: str = "last_k"
    layerwise_opd_last_k: int = 64
    layerwise_opd_first_k: int = 64
    layerwise_opd_student_layers: str = "8,14,20"
    layerwise_opd_teacher_layers: str = "8,14,20"
    layerwise_opd_teacher_target: str = "layerwise"
    layerwise_opd_vocab_mapping: Optional[str] = None
    layerwise_opd_jacobian_backward: bool = False
    layerwise_opd_student_jlens_checkpoint: Optional[str] = None
    layerwise_opd_gradient_routing: str = "raw"
    layerwise_opd_readout: str = "raw"
    layerwise_opd_detach_lm_head: bool = False
    layerwise_opd_student_affine_checkpoint: Optional[str] = None
    layerwise_opd_teacher_affine_checkpoint: Optional[str] = None
    layerwise_opd_diagnostics: bool = False
    layerwise_opd_diagnostics_interval: int = 25
    layerwise_opd_online_jcr_norm_chunk_max_tokens: int = 0
    layerwise_opd_token_chunk_size: int = 32
    layerwise_opd_park_teacher_repr_on_cpu: str = "auto"
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    rollout_n: int = 1  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        if self.loss_agg_mode not in VALID_LOSS_AGG_MODES:
            raise ValueError(
                f"Invalid loss_agg_mode: {self.loss_agg_mode}. "
                f"Expected one of {VALID_LOSS_AGG_MODES}."
            )

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
