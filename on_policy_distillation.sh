#!/bin/bash

set -x

# Configure logging when running outside SBATCH.
if [ -z "$SLURM_JOB_ID" ]; then
    # Create the log directory and file for local runs.
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    # Resource-aware launchers provide LOG_FILE so the primary log has the
    # experiment's method/batch/budget name rather than a bare timestamp.
    LOG_FILE=${LOG_FILE:-"${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"}
    # Mirror output to both terminal and log file.
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

# Keep the historical standalone behavior, but let resource-aware launchers
# avoid tearing down another user's Ray cluster.
export RAY_STOP_BEFORE_RUN=${RAY_STOP_BEFORE_RUN:-True}
if [ "$RAY_STOP_BEFORE_RUN" = "True" ]; then
    ray stop --force
else
    echo "Skipping ray stop (RAY_STOP_BEFORE_RUN=False)"
fi
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTHONUNBUFFERED=1
# Set to 1 to log mem/actor_update_* metrics (peak GPU during update_policy; compare rep-only vs top-k)
export ACTOR_UPDATE_MEM_PROFILE=0
export PROJECT_NAME=${PROJECT_NAME:-OPTD}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export ADV_ESTIMATOR=token_reward_direct
# export ADV_ESTIMATOR=token_reward_direct_plus_grpo
# export ADV_ESTIMATOR=token_grpo
# export ADV_ESTIMATOR=grpo
export GRPO_OUTCOME_WEIGHT=1.0
# export ADV_ESTIMATOR=token_grpo
# Swanlab setting used to continue exp  
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="jri5qia6iy67v7su0zjsv"


# DeepMath-103K
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-16384}  # TODO: 31744 /15360 / 7168 / 3072 / 5120
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-15360} # TODO: 15360 / 7168 / 3072
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
export FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}
# Formal-run default: persist and validate every 25 steps.
# Historical launchers may override.
export SAVE_FREQ=${SAVE_FREQ:-25}
# Formal training defaults to checkpoint-only; benchmark evaluation runs as a
# separate job. Set TEST_FREQ explicitly for a diagnostic validation cadence.
export TEST_FREQ=${TEST_FREQ:-0}
export TRAINER_LOGGERS=${TRAINER_LOGGERS:-'[console,wandb]'}
# Validation samples are logged as W&B Table artifacts.  Keep the historical
# default, but let formal launchers disable only that fragile artifact path
# while retaining normal W&B scalar metrics.
export TRAINER_LOG_VAL_GENERATIONS=${TRAINER_LOG_VAL_GENERATIONS:-2}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export VAL_ONLY=${VAL_ONLY:-False}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATA_SAMPLER_CLASS_PATH=${DATA_SAMPLER_CLASS_PATH:-}
export DATA_SAMPLER_CLASS_NAME=${DATA_SAMPLER_CLASS_NAME:-}
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-False}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8} # TODO: 1 / 8 / 16 / 32 / 64 (default 64)
# Keep rollout collection and PPO optimization independently configurable.
# Defaults preserve the historical one-rollout/one-PPO-mini-batch behavior.
# With e.g. ROLLOUT_PROMPT_BATCH_SIZE=16 and PPO_PROMPT_MINI_BATCH_SIZE=8,
# each rank receives two local PPO mini-batches from one larger rollout.
export ROLLOUT_PROMPT_BATCH_SIZE=${ROLLOUT_PROMPT_BATCH_SIZE:-${MINI_BATCH_SIZE}}
export PPO_PROMPT_MINI_BATCH_SIZE=${PPO_PROMPT_MINI_BATCH_SIZE:-${MINI_BATCH_SIZE}}
export SOURCE_HOMOGENEOUS_BATCHES=${SOURCE_HOMOGENEOUS_BATCHES:-False}
# Per-GPU micro-batch for PPO (when dynamic_bsz packs by tokens this is mostly a
# validation/legacy knob; keep overridable for fixed-bsz / assert alignment).
# With n=2 and 4 GPUs, normalized mini = PMBS*2/4; micro must divide that.
# e.g. PMBS=16 → local mini=8 → micro ∈ {1,2,4,8} (16 would fail).
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
if [ "${ROLLOUT_PROMPT_BATCH_SIZE}" = "${PPO_PROMPT_MINI_BATCH_SIZE}" ]; then
    export BATCH_SIZE_TAG="mbs_${MINI_BATCH_SIZE}"
else
    export BATCH_SIZE_TAG="rbs_${ROLLOUT_PROMPT_BATCH_SIZE}-pmbs_${PPO_PROMPT_MINI_BATCH_SIZE}"
fi
export TEMPERATURE=${TEMPERATURE:-1.0} # TODO: 0.6 / 0.8 / 1.0 / 1.2 (default 1.0)
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0} # Teacher logits temperature (default 1.0, no scaling)
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0} # TODO: 1.0 / 1.1 / 1.2 (default 1.0, no penalty)
export N_RESPONSES=${N_RESPONSES:-2} # TODO: 4 / 8 / 16 / 32 (default: 8)
export VAL_N_RESPONSES=${VAL_N_RESPONSES:-16}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16} # 0 represents no top-k sampling
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"} # "only_stu" or "only_tch" or "intersection" or "union" or "union-intersection"
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"} # "student_p" or "teacher_p" or "none"
export ACTOR_LEARNING_RATE=${ACTOR_LEARNING_RATE:-1e-5}
export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.03}
export USE_KL=${USE_KL:-False} # TODO: True / False (default False)
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False} # TODO: True / False (default False)
export MODEL_DTYPE=${MODEL_DTYPE:-fp32} # actor/ref/critic fsdp_config.model_dtype: fp32 or bfloat16
export IS_PLOT=${IS_PLOT:-True} # TODO: True / False (default False)
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"} # TODO: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean" / "seq-mean-token-sum-norm" (default "token-mean")
# Top-K OPD backward is VRAM-heavy on long responses; activation offload can also trigger errors.
export ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-False}
# Teacher/reward scoring micro-batch; long-seq JAlign often needs 1 to avoid OOM.
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-24}

# Representation-level distillation (optional, can combine with OPD).
# Quick start for rep-only: bash rep_distillation.sh
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
# Rep token positions: last | all | last_k | first_k
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-first_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-32}
export REP_DISTILLATION_FIRST_K=${REP_DISTILLATION_FIRST_K:-50}
# Rep transformer layers: last | all | even | odd
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-last}

# JLens latent alignment (+ final OPD when REP_DISTILLATION_ONLY=False).
# jalign_loss_type: opd = J-OPD KL-reward score-function (↔ LayerwiseOPD);
#                   kl  = T^2 KL ablation (↔ DirectSoftmax);
#                   mse = L2 MSE on J h (↔ OPRD).
# jalign_student_transport: frozen | periodic_refresh | stochastic_low_rank |
# conditional_residual | identity_adapter.
# jalign_gradient_routing: jacobian | identity_credit.  The latter preserves
# the J h forward target but uses identity surrogate credit to the actor state.
# jalign_opd_aggregation: equal_weight | legacy_additive.
export USE_JLENS_DISTILLATION=${USE_JLENS_DISTILLATION:-False}
export JALIGN_LOSS_TYPE=${JALIGN_LOSS_TYPE:-opd}
export JALIGN_OPD_SUPPORT=${JALIGN_OPD_SUPPORT:-final_policy}
export JALIGN_STUDENT_TRANSPORT=${JALIGN_STUDENT_TRANSPORT:-frozen}
export JALIGN_GRADIENT_ROUTING=${JALIGN_GRADIENT_ROUTING:-jacobian}
export JALIGN_OPD_AGGREGATION=${JALIGN_OPD_AGGREGATION:-equal_weight}
export JFLOW_COEF=${JFLOW_COEF:-1.0}
export JFLOW_TOKEN_OFFSET=${JFLOW_TOKEN_OFFSET:-8}
export JALIGN_STOCH_RANK=${JALIGN_STOCH_RANK:-8}
export JALIGN_STOCH_EPS=${JALIGN_STOCH_EPS:-1e-3}
export JALIGN_STOCH_PROBE=${JALIGN_STOCH_PROBE:-rademacher}
export JALIGN_CONDITIONAL_RANK=${JALIGN_CONDITIONAL_RANK:-1}
export JALIGN_CONDITIONAL_LR=${JALIGN_CONDITIONAL_LR:-2e-3}
export JALIGN_CONDITIONAL_TRANSPORT_COEF=${JALIGN_CONDITIONAL_TRANSPORT_COEF:-1.0}
export JALIGN_IDENTITY_ADAPTER_RANK=${JALIGN_IDENTITY_ADAPTER_RANK:-8}
export JALIGN_IDENTITY_ADAPTER_LR=${JALIGN_IDENTITY_ADAPTER_LR:-2e-3}
export JALIGN_IDENTITY_ADAPTER_TRANSPORT_COEF=${JALIGN_IDENTITY_ADAPTER_TRANSPORT_COEF:-1.0}
export JALIGN_REFRESH_INTERVAL=${JALIGN_REFRESH_INTERVAL:-25}
export JALIGN_REFRESH_PROBE_DATA=${JALIGN_REFRESH_PROBE_DATA:-datasets/test_data/DAPO/train.json}
export JALIGN_REFRESH_RESPONSE_TOKEN_DATA=${JALIGN_REFRESH_RESPONSE_TOKEN_DATA:-null}
export JALIGN_REFRESH_TEXT_KEY=${JALIGN_REFRESH_TEXT_KEY:-prompt}
export JALIGN_REFRESH_NUM_PROMPTS=${JALIGN_REFRESH_NUM_PROMPTS:-100}
export JALIGN_REFRESH_DIM_BATCH=${JALIGN_REFRESH_DIM_BATCH:-64}
export JALIGN_REFRESH_MAX_SEQ_LEN=${JALIGN_REFRESH_MAX_SEQ_LEN:-128}
export JALIGN_REFRESH_MAX_RESPONSE_POSITIONS=${JALIGN_REFRESH_MAX_RESPONSE_POSITIONS:-2000}
export JALIGN_REFRESH_SKIP_FIRST=${JALIGN_REFRESH_SKIP_FIRST:-16}
export JALIGN_REFRESH_SEED=${JALIGN_REFRESH_SEED:-42}
export JALIGN_REFRESH_CUDA_VISIBLE_DEVICES=${JALIGN_REFRESH_CUDA_VISIBLE_DEVICES:-null}
# For J-OPD, this is the weight of *each* intermediate J-layer relative to
# final OPD; 1.0 gives equal weight to final OPD and every selected J-layer.
export JALIGN_COEF=${JALIGN_COEF:-1.0}
export JALIGN_TEMPERATURE=${JALIGN_TEMPERATURE:-1.0}
export JALIGN_POSITIONS=${JALIGN_POSITIONS:-last_k}
export JALIGN_LAST_K=${JALIGN_LAST_K:-64}
export JALIGN_FIRST_K=${JALIGN_FIRST_K:-64}
export JALIGN_STUDENT_LAYERS=${JALIGN_STUDENT_LAYERS:-8,14,20}
export JALIGN_TEACHER_LAYERS=${JALIGN_TEACHER_LAYERS:-8,14,20}
export STUDENT_JLENS_CHECKPOINT=${STUDENT_JLENS_CHECKPOINT:-outputs/jlens_v1/student}
export TEACHER_JLENS_CHECKPOINT=${TEACHER_JLENS_CHECKPOINT:-outputs/jlens_v1/teacher}
export JALIGN_VOCAB_MAPPING=${JALIGN_VOCAB_MAPPING:-null}
# Optional override; periodic refresh launcher sets this to include hf_model.
export ACTOR_CHECKPOINT_SAVE_CONTENTS=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra]}
export ACTOR_CHECKPOINT_LOAD_CONTENTS=${ACTOR_CHECKPOINT_LOAD_CONTENTS:-[model,optimizer,extra]}

# No-JLens control: apply final norm + LM head directly to selected intermediate
# states, then align the resulting softmax distributions.  Keep its defaults
# matched to JAlign for a one-variable ablation.
export USE_DIRECT_SOFTMAX_DISTILLATION=${USE_DIRECT_SOFTMAX_DISTILLATION:-False}
export DIRECT_SOFTMAX_COEF=${DIRECT_SOFTMAX_COEF:-1.0}
export DIRECT_SOFTMAX_TEMPERATURE=${DIRECT_SOFTMAX_TEMPERATURE:-1.0}
export DIRECT_SOFTMAX_POSITIONS=${DIRECT_SOFTMAX_POSITIONS:-last_k}
export DIRECT_SOFTMAX_LAST_K=${DIRECT_SOFTMAX_LAST_K:-64}
export DIRECT_SOFTMAX_FIRST_K=${DIRECT_SOFTMAX_FIRST_K:-64}
export DIRECT_SOFTMAX_STUDENT_LAYERS=${DIRECT_SOFTMAX_STUDENT_LAYERS:-8,14,20}
export DIRECT_SOFTMAX_TEACHER_LAYERS=${DIRECT_SOFTMAX_TEACHER_LAYERS:-8,14,20}
export DIRECT_SOFTMAX_VOCAB_MAPPING=${DIRECT_SOFTMAX_VOCAB_MAPPING:-null}

# Layerwise OPD: top-k detached pseudo-reward / score-function update applied
# to intermediate LM-head readouts.  ``final_policy`` shares the ordinary
# final-policy top-k support; ``self_student`` freezes each layer's own
# student top-k support at the rollout-policy snapshot.
# This is the weight of *each* intermediate layer relative to final OPD;
# 1.0 gives equal per-layer weights in the unified objective.
export USE_LAYERWISE_OPD_DISTILLATION=${USE_LAYERWISE_OPD_DISTILLATION:-False}
export LAYERWISE_OPD_COEF=${LAYERWISE_OPD_COEF:-1.0}
export LAYERWISE_OPD_ADAPTIVE=${LAYERWISE_OPD_ADAPTIVE:-False}
export LAYERWISE_OPD_ADAPTIVE_EMA_BETA=${LAYERWISE_OPD_ADAPTIVE_EMA_BETA:-0.9}
export LAYERWISE_OPD_ADAPTIVE_TEMPERATURE=${LAYERWISE_OPD_ADAPTIVE_TEMPERATURE:-1.0}
export LAYERWISE_OPD_OBJECTIVE=${LAYERWISE_OPD_OBJECTIVE:-absolute}
export LAYERWISE_OPD_DELTA_ALIGN=${LAYERWISE_OPD_DELTA_ALIGN:-False}
export LAYERWISE_OPD_REVERSE_TRANSITION_KL=${LAYERWISE_OPD_REVERSE_TRANSITION_KL:-False}
export LAYERWISE_OPD_WEIGHT_ONLY=${LAYERWISE_OPD_WEIGHT_ONLY:-False}
export LAYERWISE_OPD_CENTER_ONLY=${LAYERWISE_OPD_CENTER_ONLY:-False}
export LAYERWISE_OPD_TRANSITION_WEIGHTS=${LAYERWISE_OPD_TRANSITION_WEIGHTS:-}
export LAYERWISE_OPD_PAIR_SELECTION=${LAYERWISE_OPD_PAIR_SELECTION:-none}
# equal_weight | legacy_additive.  The latter reproduces the early objective
# L_final + coef * mean_l(L_l), where the intermediate loss is pre-averaged.
export LAYERWISE_OPD_AGGREGATION=${LAYERWISE_OPD_AGGREGATION:-equal_weight}
export LAYERWISE_OPD_SAMPLE_ONE_LAYER=${LAYERWISE_OPD_SAMPLE_ONE_LAYER:-False}
export LAYERWISE_OPD_TEMPERATURE=${LAYERWISE_OPD_TEMPERATURE:-1.0}
export LAYERWISE_OPD_TOP_K=${LAYERWISE_OPD_TOP_K:-16}
export LAYERWISE_OPD_SUPPORT=${LAYERWISE_OPD_SUPPORT:-final_policy}
export LAYERWISE_OPD_POSITIONS=${LAYERWISE_OPD_POSITIONS:-last_k}
export LAYERWISE_OPD_LAST_K=${LAYERWISE_OPD_LAST_K:-2000}
export LAYERWISE_OPD_FIRST_K=${LAYERWISE_OPD_FIRST_K:-64}
export LAYERWISE_OPD_STUDENT_LAYERS=${LAYERWISE_OPD_STUDENT_LAYERS:-8,14,20}
export LAYERWISE_OPD_TEACHER_LAYERS=${LAYERWISE_OPD_TEACHER_LAYERS:-8,14,20}
export LAYERWISE_OPD_TEACHER_TARGET=${LAYERWISE_OPD_TEACHER_TARGET:-layerwise}
export LAYERWISE_OPD_TOKEN_CHUNK_SIZE=${LAYERWISE_OPD_TOKEN_CHUNK_SIZE:-32}
export LAYERWISE_OPD_VOCAB_MAPPING=${LAYERWISE_OPD_VOCAB_MAPPING:-null}
# auto: park teacher hidden on CPU only when >=8 layerwise layers (AllBlocks).
export LAYERWISE_OPD_PARK_TEACHER_REPR_ON_CPU=${LAYERWISE_OPD_PARK_TEACHER_REPR_ON_CPU:-auto}
# Keep LayerwiseOPD logits unchanged while applying the student JLens only to
# the intermediate backward path.  Requires a lens containing the selected layers.
export LAYERWISE_OPD_JACOBIAN_BACKWARD=${LAYERWISE_OPD_JACOBIAN_BACKWARD:-False}
export LAYERWISE_OPD_STUDENT_JLENS_CHECKPOINT=${LAYERWISE_OPD_STUDENT_JLENS_CHECKPOINT:-null}
# raw | frozen_jlens | online_jcr.  The legacy boolean above remains a frozen
# JLens-only compatibility alias and must stay False for Online-JCR.
export LAYERWISE_OPD_GRADIENT_ROUTING=${LAYERWISE_OPD_GRADIENT_ROUTING:-raw}
# raw | frozen_affine. This controls the forward intermediate readout and is
# independent from gradient-routing modes such as frozen_jlens/online_jcr.
export LAYERWISE_OPD_READOUT=${LAYERWISE_OPD_READOUT:-raw}
export LAYERWISE_OPD_DETACH_LM_HEAD=${LAYERWISE_OPD_DETACH_LM_HEAD:-False}
export LAYERWISE_OPD_STUDENT_AFFINE_CHECKPOINT=${LAYERWISE_OPD_STUDENT_AFFINE_CHECKPOINT:-null}
export LAYERWISE_OPD_TEACHER_AFFINE_CHECKPOINT=${LAYERWISE_OPD_TEACHER_AFFINE_CHECKPOINT:-null}
# Diagnostic-only. Adds scalar distribution/cotangent logging and snapshots
# local FSDP parameter shards around optimizer.step to measure the true update.
# It does not enter the objective and remains off for formal runs by default.
export LAYERWISE_OPD_DIAGNOSTICS=${LAYERWISE_OPD_DIAGNOSTICS:-False}
export LAYERWISE_OPD_DIAGNOSTICS_INTERVAL=${LAYERWISE_OPD_DIAGNOSTICS_INTERVAL:-25}
export LAYERWISE_OPD_ONLINE_JCR_NORM_CHUNK_MAX_TOKENS=${LAYERWISE_OPD_ONLINE_JCR_NORM_CHUNK_MAX_TOKENS:-0}

# Attention distillation (optional)
export USE_ATT_DISTILLATION=${USE_ATT_DISTILLATION:-False}
export ATT_DISTILLATION_COEF=${ATT_DISTILLATION_COEF:-1.0}
export ATT_DISTILLATION_LAYERS=${ATT_DISTILLATION_LAYERS:-last}
export ATT_DISTILLATION_POSITIONS=${ATT_DISTILLATION_POSITIONS:-first_k}
export ATT_DISTILLATION_LAST_K=${ATT_DISTILLATION_LAST_K:-32}
export ATT_DISTILLATION_FIRST_K=${ATT_DISTILLATION_FIRST_K:-50}
export ATT_DISTILLATION_MAX_KEY_LEN=${ATT_DISTILLATION_MAX_KEY_LEN:-4096}
export ATT_DISTILLATION_LOSS=${ATT_DISTILLATION_LOSS:-kl}
export ATT_DISTILLATION_TEMPERATURE=${ATT_DISTILLATION_TEMPERATURE:-1.0}

# TODO: qwen3_1p7b_base / qwen3_1p7b / llama31_8b_base / llama31_8b_inst / qwen3_8b_base / qwen3_8b / qwen25_1p5b_base / qwen25_1p5b_inst / qwen25_7b_base / qwen25_7b_inst / qwen25_math_7b_base / qwen25_math_7b_inst / qwen25_math_1p5b_base / qwen25_math_1p5b_inst / distill_r1_1p5b / olmo2_1124_7b_base / olmo2_1124_7b_sft / olmo2_1124_7b_inst / llama32_3b_inst
# export EXPERIMENT_NAME=grpo_${TASK}_llama31_tulu3_8b_sft_8k-T_${TEMPERATURE}-n_${N_RESPONSES}-kl_${USE_KL}-mbs_${MINI_BATCH_SIZE}-${REWARD_TYPE}-$(date +%Y-%m-%d_%H-%M-%S)

# export TRAIN_DATASET=datasets/DAPO-Math-17k/data/dapo-math-17k-10percent.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/OpenThoughts3_opd.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/sampled_complement_30k.parquet
# export TRAIN_DATASET=datasets/DeepMath-103K/verl_format/train_filtered_sampled.parquet
export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
# export TRAIN_DATASET=${DATA_DIR}/DeepMath-103K/train_filtered_level6.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/data/math-00000-of-00001.parquet
# export TRAIN_DATASET=datasets/Skywork-OR1-RL-Data/filtered/math-1p5b-filtered-diff-max8.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet
# export TRAIN_DATASET=datasets/skywork/train_7b_math.parquet
# export TRAIN_DATASET=datasets/DAPO-Math-17k-Processed/DAPO-Math_part2.parquet
# export TRAIN_DATASET=datasets/OpenThoughts3-1.2M/verl_format/train.parquet
# export TRAIN_DATASET_NAME=DAPO-Math-17k
# export TRAIN_DATASET_NAME=POLARIS-4B-S1
# export TRAIN_DATASET_NAME=Skywork-OR1-RL-Data
# export TRAIN_DATASET_NAME=DAPO-Math-17k-1percent
# export TRAIN_DATASET_NAME=DeepMath-103K-filtered-sampled_test
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}
# export TRAIN_DATASET_NAME=DAPO-Math-17k-10percent
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-opd
# export TRAIN_DATASET_NAME=OpenThoughts3-1.2M-30k

export TEST_DATA_DIR=datasets/test_data
# TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/$TASK/train_${SAMPLE_SIZE}.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
# TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME24/test.parquet"]}
TEST_DATASET=${TEST_DATASET:-${TEST_FILE:-["$TEST_DATA_DIR/AMC23/test.parquet"]}}
# TEST_DATASET=${TEST_FILE:-["$DATA_DIR/AIME24/test.parquet","$DATA_DIR/AIME25/test.parquet","$DATA_DIR/AMC23/test.parquet","$DATA_DIR/MATH-500/test.parquet","$DATA_DIR/Minerva/test.parquet","$DATA_DIR/Olympiad-Bench/test.parquet"]}

# TODO:
# export ACTOR_MODEL_PATH=model/qwen3-1.7b-math-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-skywork
# export ACTOR_MODEL_PATH=model/DS-1.5B-sft-ds-7b
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B-RL
# export ACTOR_MODEL_PATH=/workspace/model/Qwen3-1.7B-SFT-DAPO-4B
# export ACTOR_MODEL_PATH=model/Qwen2.5-Math-1.5B
# export ACTOR_MODEL_PATH=${MODEL_DIR}/Qwen3-4B
# export ACTOR_MODEL_PATH=${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=${CKPT_DIR}/token_reward_direct_..._global_step_10_oprd_r8
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}
# export ACTOR_MODEL_PATH=${CKPT_DIR}/Qwen3-1.7B-Base-SFT-OpenThought3-4B/checkpoint-400
# export ACTOR_MODEL_PATH=model/JustRL-DeepSeek-1.5B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-SFT
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-OpenThought3-4B/checkpoint-1800
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-Base-SFT-DeepMath-4B
# export ACTOR_MODEL_PATH=model/Qwen3-1.7B-sft/checkpoint-6000
# export ACTOR_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export ACTOR_MODEL_PATH=model/DS-1.5B-SFT
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
# export REWARD_MODEL_PATH=model/Qwen3-4B
# export REWARD_MODEL_PATH=model/Qwen3-4B-grpo
# export REWARD_MODEL_PATH=model/Qwen3-1.7B
# export REWARD_MODEL_PATH=model/OpenMath-Nemotron-1.5B
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-7B
# export REWARD_MODEL_PATH=model/Qwen3-4B-Non-Thinking-RL-Math
# export REWARD_MODEL_PATH=model/Skywork-OR1-Math-7B
# export REWARD_MODEL_PATH=model/Polaris-4B-Preview
# export REWARD_MODEL_PATH=model/DeepSeek-R1-Distill-Qwen-14B
# export REWARD_MODEL_PATH=${MODEL_DIR}/JustRL-DeepSeek-1.5B  
# export REWARD_MODEL_PATH=${MODEL_DIR}/JustRL-DeepSeek-1.5B
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}
# export REWARD_MODEL_PATH=${MODEL_DIR}/Phi-4-mini-reasoning

# Optional explicit data-source routing to a second same-lineage teacher.
export SECONDARY_REWARD_MODEL_PATH=${SECONDARY_REWARD_MODEL_PATH:-null}
export TERTIARY_REWARD_MODEL_PATH=${TERTIARY_REWARD_MODEL_PATH:-null}
export ALL_TEACHER_LOSS_MEAN=${ALL_TEACHER_LOSS_MEAN:-False}
export EXPERTALIGN_ENABLED=${EXPERTALIGN_ENABLED:-False}
export TEACHER_BASE_MODEL_PATH=${TEACHER_BASE_MODEL_PATH:-null}
export EXPERTALIGN_WEIGHTING=${EXPERTALIGN_WEIGHTING:-cosine}
export EXPERTALIGN_ALIGNMENT_EPSILON=${EXPERTALIGN_ALIGNMENT_EPSILON:-1e-6}
export EXPERTALIGN_LAYERWISE_PER_LAYER=${EXPERTALIGN_LAYERWISE_PER_LAYER:-False}
export PRIMARY_TEACHER_JLENS_CHECKPOINT=${PRIMARY_TEACHER_JLENS_CHECKPOINT:-null}
export SECONDARY_TEACHER_JLENS_CHECKPOINT=${SECONDARY_TEACHER_JLENS_CHECKPOINT:-null}
export PRIMARY_TEACHER_DATA_SOURCES=${PRIMARY_TEACHER_DATA_SOURCES:-[]}
export SECONDARY_TEACHER_DATA_SOURCES=${SECONDARY_TEACHER_DATA_SOURCES:-[]}
export TERTIARY_TEACHER_DATA_SOURCES=${TERTIARY_TEACHER_DATA_SOURCES:-[]}
export NO_PREASSISTANT_EOS_DATA_SOURCES=${NO_PREASSISTANT_EOS_DATA_SOURCES:-[]}

# The historical math-only reward remains the default. Mixed code runs set
# CUSTOM_REWARD_FUNCTION_PATH=null and must provide a remote sandbox URL.
export CUSTOM_REWARD_FUNCTION_PATH=${CUSTOM_REWARD_FUNCTION_PATH:-verl/verl/utils/reward_score/ttrl_math/__init__.py}
export CUSTOM_REWARD_FUNCTION_NAME=${CUSTOM_REWARD_FUNCTION_NAME:-reward_func}
export SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-null}
export SANDBOX_FUSION_MAX_CONCURRENT=${SANDBOX_FUSION_MAX_CONCURRENT:-64}
export SANDBOX_FUSION_MEMORY_LIMIT_MB=${SANDBOX_FUSION_MEMORY_LIMIT_MB:-1024}
export SKIP_RULE_REWARD_WHEN_RM_SCORES=${SKIP_RULE_REWARD_WHEN_RM_SCORES:-False}
export CODE_REWARD_CONTINUOUS=${CODE_REWARD_CONTINUOUS:-True}
export CODE_REWARD_ALL_PASSED=${CODE_REWARD_ALL_PASSED:-False}
export CODE_REWARD_LCB_COMPAT=${CODE_REWARD_LCB_COMPAT:-False}
export CHAT_TEMPLATE_PROFILE=${CHAT_TEMPLATE_PROFILE:-null}

export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=${PROJECT_PATH:-./outputs}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
export RUN_LABEL=${RUN_LABEL:-OPD}
# Allow callers to pin an existing run dir for resume (do not regenerate timestamp).
if [ -z "${CKPT_PATH:-}" ]; then
    export CKPT_PATH=${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-${BATCH_SIZE_TAG}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)-${RUN_LABEL}
fi
export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-~/.cache/outlines/$(uuidgen)}
export NCCL_DEBUG=WARN

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1

# resume_mode: auto | disable | resume_path
export RESUME_MODE=${RESUME_MODE:-auto}
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}

if [ -z "${EXPERIMENT_NAME:-}" ]; then
    export EXPERIMENT_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-${BATCH_SIZE_TAG}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)-${RUN_LABEL}
fi

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO"
fi

DEFAULT_MAX_TOKEN_LEN=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
# The actor log-prob/update micro-batch can be smaller than vLLM's rollout
# scheduler budget. Keeping these independent avoids changing generation
# batching merely to accommodate the full-vocabulary top-k log-prob readout.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$DEFAULT_MAX_TOKEN_LEN}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$PPO_MAX_TOKEN_LEN_PER_GPU}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_TOKEN_LEN}
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "LOG_PROB_MAX_TOKEN_LEN_PER_GPU: $LOG_PROB_MAX_TOKEN_LEN_PER_GPU"


if [ "${RAY_START_BEFORE_RUN:-True}" = "True" ]; then
    ray start --head ${RAY_START_ARGS:-} || exit 1
else
    echo "Skipping ray start (RAY_START_BEFORE_RUN=False)"
fi
sleep 5
ray status --address="${RAY_ADDRESS:-auto}" >/dev/null || exit 1

# Optional diagnostic-only fail-fast gate used by the routed multi-teacher
# launchers.  It verifies both the driver and an actual Ray worker import the
# intended checkout and can dispatch every validation source.  This does not
# enter the training objective.
if [ -n "${MULTITEACHER_PREFLIGHT_SCRIPT:-}" ]; then
    if [ -z "${OPRD_EXPECTED_REPO_ROOT:-}" ] || [ -z "${MULTITEACHER_VALIDATION_FILES:-}" ] || [ -z "${MULTITEACHER_SANDBOX_URL:-}" ]; then
        echo "ERROR: incomplete multi-teacher preflight configuration" >&2
        exit 1
    fi
    PREFLIGHT_VALIDATION_ARGS=()
    for validation_file in ${MULTITEACHER_VALIDATION_FILES}; do
        PREFLIGHT_VALIDATION_ARGS+=(--validation-file "$validation_file")
    done
    python3 "$MULTITEACHER_PREFLIGHT_SCRIPT" \
        --expected-root "$OPRD_EXPECTED_REPO_ROOT" \
        --ray-address "${RAY_ADDRESS:-auto}" \
        --sandbox-url "$MULTITEACHER_SANDBOX_URL" \
        "${PREFLIGHT_VALIDATION_ARGS[@]}" || exit 1
fi

# Record physical-GPU telemetry for every on-policy experiment by default.
# The recorder terminates automatically when this launcher exits.
GPU_MONITOR_ENABLED=${GPU_MONITOR_ENABLED:-True}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-10}
if [ "$GPU_MONITOR_ENABLED" = "True" ]; then
    GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-${CKPT_PATH}/gpu_telemetry.csv}"
    mkdir -p "$(dirname "$GPU_MONITOR_LOG")"
    bash "$(dirname "$0")/scripts/monitor_gpu_state.sh" \
        --log "$GPU_MONITOR_LOG" \
        --gpus "${CUDA_VISIBLE_DEVICES:-}" \
        --pid "$$" \
        --interval "$GPU_MONITOR_INTERVAL" &
    echo "GPU telemetry: $GPU_MONITOR_LOG (every ${GPU_MONITOR_INTERVAL}s)"
fi

if [ "$SOURCE_HOMOGENEOUS_BATCHES" = "True" ]; then
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_PATH=${SOURCE_HOMOGENEOUS_SAMPLER_CLASS_PATH:-pkg://verl.utils.dataset.source_homogeneous_sampler}
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_NAME=${SOURCE_HOMOGENEOUS_SAMPLER_CLASS_NAME:-SourceHomogeneousBatchSampler}
else
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_PATH=null
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_NAME=null
fi

if [ "${DATA_SAMPLING_MODE:-}" = "balanced_cycle" ]; then
    if [ "$SOURCE_HOMOGENEOUS_BATCHES" = "True" ]; then
        echo "balanced_cycle requires SOURCE_HOMOGENEOUS_BATCHES=False" >&2
        exit 2
    fi
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_PATH=pkg://verl.utils.dataset.balanced_cycle_sampler
    SOURCE_HOMOGENEOUS_SAMPLER_CLASS_NAME=BalancedCycleSampler
fi

REWARD_BATCH_ARGS=()
if [[ -n "${REWARD_USE_DYNAMIC_BSZ:-}" ]]; then
    REWARD_BATCH_ARGS+=("reward_model.use_dynamic_bsz=$REWARD_USE_DYNAMIC_BSZ")
fi
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${ROLLOUT_PROMPT_BATCH_SIZE}*${PARALLEL_SIZE})) \
    +data.sampler.class_path=${SOURCE_HOMOGENEOUS_SAMPLER_CLASS_PATH:-null} \
    +data.sampler.class_name=${SOURCE_HOMOGENEOUS_SAMPLER_CLASS_NAME:-null} \
    +data.sampler.source_homogeneous_num_batches=${TOTAL_TRAINING_STEPS} \
    +data.sampler.balanced_cycle_num_batches=${TOTAL_TRAINING_STEPS} \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    ${DATA_SAMPLER_CLASS_PATH:+data.sampler.class_path=$DATA_SAMPLER_CLASS_PATH} \
    ${DATA_SAMPLER_CLASS_NAME:+data.sampler.class_name=$DATA_SAMPLER_CLASS_NAME} \
    +data.chat_template_profile=$CHAT_TEMPLATE_PROFILE \
    "+data.no_preassistant_eos_data_sources=$NO_PREASSISTANT_EOS_DATA_SOURCES" \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-False} \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
    actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LEARNING_RATE \
    ${USE_TORCH_COMPILE_OVERRIDE:+actor_rollout_ref.actor.use_torch_compile=$USE_TORCH_COMPILE_OVERRIDE} \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_PROMPT_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=$ENTROPY_FROM_LOGITS_WITH_CHUNKING \
    +actor_rollout_ref.actor.use_rep_distillation=$USE_REP_DISTILLATION \
    +actor_rollout_ref.actor.rep_distillation_coef=$REP_DISTILLATION_COEF \
    +actor_rollout_ref.actor.rep_distillation_crossfade_steps=${REP_DISTILLATION_CROSSFADE_STEPS:-0} \
    +actor_rollout_ref.actor.rep_distillation_only=$REP_DISTILLATION_ONLY \
    +actor_rollout_ref.actor.rep_distillation_positions=$REP_DISTILLATION_POSITIONS \
    +actor_rollout_ref.actor.rep_distillation_last_k=$REP_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.rep_distillation_first_k=$REP_DISTILLATION_FIRST_K \
    "+actor_rollout_ref.actor.rep_distillation_layers='$REP_DISTILLATION_LAYERS'" \
    +actor_rollout_ref.actor.use_jlens_distillation=$USE_JLENS_DISTILLATION \
    +actor_rollout_ref.actor.jalign_loss_type=$JALIGN_LOSS_TYPE \
    +actor_rollout_ref.actor.jalign_opd_support=$JALIGN_OPD_SUPPORT \
    +actor_rollout_ref.actor.jalign_student_transport=$JALIGN_STUDENT_TRANSPORT \
    +actor_rollout_ref.actor.jalign_gradient_routing=$JALIGN_GRADIENT_ROUTING \
    +actor_rollout_ref.actor.jalign_opd_aggregation=$JALIGN_OPD_AGGREGATION \
    +actor_rollout_ref.actor.jflow_coef=$JFLOW_COEF \
    +actor_rollout_ref.actor.jflow_token_offset=$JFLOW_TOKEN_OFFSET \
    +actor_rollout_ref.actor.jalign_stoch_rank=$JALIGN_STOCH_RANK \
    +actor_rollout_ref.actor.jalign_stoch_eps=$JALIGN_STOCH_EPS \
    +actor_rollout_ref.actor.jalign_stoch_probe=$JALIGN_STOCH_PROBE \
    +actor_rollout_ref.actor.jalign_conditional_rank=$JALIGN_CONDITIONAL_RANK \
    +actor_rollout_ref.actor.jalign_conditional_lr=$JALIGN_CONDITIONAL_LR \
    +actor_rollout_ref.actor.jalign_conditional_transport_coef=$JALIGN_CONDITIONAL_TRANSPORT_COEF \
    +actor_rollout_ref.actor.jalign_identity_adapter_rank=$JALIGN_IDENTITY_ADAPTER_RANK \
    +actor_rollout_ref.actor.jalign_identity_adapter_lr=$JALIGN_IDENTITY_ADAPTER_LR \
    +actor_rollout_ref.actor.jalign_identity_adapter_transport_coef=$JALIGN_IDENTITY_ADAPTER_TRANSPORT_COEF \
    +actor_rollout_ref.actor.jalign_refresh_interval=$JALIGN_REFRESH_INTERVAL \
    +actor_rollout_ref.actor.jalign_refresh_probe_data=$JALIGN_REFRESH_PROBE_DATA \
    +actor_rollout_ref.actor.jalign_refresh_response_token_data=$JALIGN_REFRESH_RESPONSE_TOKEN_DATA \
    +actor_rollout_ref.actor.jalign_refresh_text_key=$JALIGN_REFRESH_TEXT_KEY \
    +actor_rollout_ref.actor.jalign_refresh_num_prompts=$JALIGN_REFRESH_NUM_PROMPTS \
    +actor_rollout_ref.actor.jalign_refresh_dim_batch=$JALIGN_REFRESH_DIM_BATCH \
    +actor_rollout_ref.actor.jalign_refresh_max_seq_len=$JALIGN_REFRESH_MAX_SEQ_LEN \
    +actor_rollout_ref.actor.jalign_refresh_max_response_positions=$JALIGN_REFRESH_MAX_RESPONSE_POSITIONS \
    +actor_rollout_ref.actor.jalign_refresh_skip_first=$JALIGN_REFRESH_SKIP_FIRST \
    +actor_rollout_ref.actor.jalign_refresh_seed=$JALIGN_REFRESH_SEED \
    +actor_rollout_ref.actor.jalign_refresh_cuda_visible_devices=$JALIGN_REFRESH_CUDA_VISIBLE_DEVICES \
    +actor_rollout_ref.actor.jalign_coef=$JALIGN_COEF \
    +actor_rollout_ref.actor.jalign_temperature=$JALIGN_TEMPERATURE \
    +actor_rollout_ref.actor.jalign_positions=$JALIGN_POSITIONS \
    +actor_rollout_ref.actor.jalign_last_k=$JALIGN_LAST_K \
    +actor_rollout_ref.actor.jalign_first_k=$JALIGN_FIRST_K \
    "+actor_rollout_ref.actor.jalign_student_layers='${JALIGN_STUDENT_LAYERS}'" \
    "+actor_rollout_ref.actor.jalign_teacher_layers='${JALIGN_TEACHER_LAYERS}'" \
    +actor_rollout_ref.actor.student_jlens_checkpoint=$STUDENT_JLENS_CHECKPOINT \
    +actor_rollout_ref.actor.teacher_jlens_checkpoint=$TEACHER_JLENS_CHECKPOINT \
    +actor_rollout_ref.actor.primary_teacher_jlens_checkpoint=$PRIMARY_TEACHER_JLENS_CHECKPOINT \
    +actor_rollout_ref.actor.secondary_teacher_jlens_checkpoint=$SECONDARY_TEACHER_JLENS_CHECKPOINT \
    +actor_rollout_ref.actor.jalign_vocab_mapping=$JALIGN_VOCAB_MAPPING \
    +actor_rollout_ref.actor.use_direct_softmax_distillation=$USE_DIRECT_SOFTMAX_DISTILLATION \
    +actor_rollout_ref.actor.pure_forward_kl=${PURE_FORWARD_KL:-False} \
    +actor_rollout_ref.actor.pure_forward_kl_readout=${PURE_FORWARD_KL_READOUT:-raw} \
    +actor_rollout_ref.actor.pure_forward_kl_teacher_model=${PURE_FORWARD_KL_TEACHER_MODEL:-null} \
    +actor_rollout_ref.actor.direct_softmax_coef=$DIRECT_SOFTMAX_COEF \
    +actor_rollout_ref.actor.direct_softmax_temperature=$DIRECT_SOFTMAX_TEMPERATURE \
    +actor_rollout_ref.actor.direct_softmax_positions=$DIRECT_SOFTMAX_POSITIONS \
    +actor_rollout_ref.actor.direct_softmax_last_k=$DIRECT_SOFTMAX_LAST_K \
    +actor_rollout_ref.actor.direct_softmax_first_k=$DIRECT_SOFTMAX_FIRST_K \
    "+actor_rollout_ref.actor.direct_softmax_student_layers='${DIRECT_SOFTMAX_STUDENT_LAYERS}'" \
    "+actor_rollout_ref.actor.direct_softmax_teacher_layers='${DIRECT_SOFTMAX_TEACHER_LAYERS}'" \
    +actor_rollout_ref.actor.direct_softmax_vocab_mapping=$DIRECT_SOFTMAX_VOCAB_MAPPING \
    +actor_rollout_ref.actor.use_layerwise_opd_distillation=$USE_LAYERWISE_OPD_DISTILLATION \
    +actor_rollout_ref.actor.expertalign_layerwise_per_layer=$EXPERTALIGN_LAYERWISE_PER_LAYER \
    +actor_rollout_ref.actor.layerwise_opd_coef=$LAYERWISE_OPD_COEF \
    +actor_rollout_ref.actor.layerwise_opd_adaptive=$LAYERWISE_OPD_ADAPTIVE \
    +actor_rollout_ref.actor.layerwise_opd_adaptive_ema_beta=$LAYERWISE_OPD_ADAPTIVE_EMA_BETA \
    +actor_rollout_ref.actor.layerwise_opd_adaptive_temperature=$LAYERWISE_OPD_ADAPTIVE_TEMPERATURE \
    +actor_rollout_ref.actor.layerwise_opd_objective=$LAYERWISE_OPD_OBJECTIVE \
    +actor_rollout_ref.actor.layerwise_opd_delta_align=$LAYERWISE_OPD_DELTA_ALIGN \
    +actor_rollout_ref.actor.layerwise_opd_reverse_transition_kl=$LAYERWISE_OPD_REVERSE_TRANSITION_KL \
    +actor_rollout_ref.actor.layerwise_opd_weight_only=$LAYERWISE_OPD_WEIGHT_ONLY \
    +actor_rollout_ref.actor.layerwise_opd_center_only=$LAYERWISE_OPD_CENTER_ONLY \
    "+actor_rollout_ref.actor.layerwise_opd_transition_weights='${LAYERWISE_OPD_TRANSITION_WEIGHTS}'" \
    +actor_rollout_ref.actor.layerwise_opd_pair_selection=$LAYERWISE_OPD_PAIR_SELECTION \
    +actor_rollout_ref.actor.trim_common_padding=${TRIM_COMMON_PADDING:-False} \
    +actor_rollout_ref.actor.layerwise_opd_aggregation=$LAYERWISE_OPD_AGGREGATION \
    +actor_rollout_ref.actor.layerwise_opd_sample_one_layer=$LAYERWISE_OPD_SAMPLE_ONE_LAYER \
    +actor_rollout_ref.actor.layerwise_opd_temperature=$LAYERWISE_OPD_TEMPERATURE \
    +actor_rollout_ref.actor.layerwise_opd_top_k=$LAYERWISE_OPD_TOP_K \
    +actor_rollout_ref.actor.layerwise_opd_support=$LAYERWISE_OPD_SUPPORT \
    +actor_rollout_ref.actor.layerwise_opd_positions=$LAYERWISE_OPD_POSITIONS \
    +actor_rollout_ref.actor.layerwise_opd_last_k=$LAYERWISE_OPD_LAST_K \
    +actor_rollout_ref.actor.layerwise_opd_first_k=$LAYERWISE_OPD_FIRST_K \
    "+actor_rollout_ref.actor.layerwise_opd_student_layers='${LAYERWISE_OPD_STUDENT_LAYERS}'" \
    "+actor_rollout_ref.actor.layerwise_opd_teacher_layers='${LAYERWISE_OPD_TEACHER_LAYERS}'" \
    +actor_rollout_ref.actor.layerwise_opd_teacher_target=$LAYERWISE_OPD_TEACHER_TARGET \
    +actor_rollout_ref.actor.layerwise_opd_token_chunk_size=$LAYERWISE_OPD_TOKEN_CHUNK_SIZE \
    "+actor_rollout_ref.actor.layerwise_opd_park_teacher_repr_on_cpu='${LAYERWISE_OPD_PARK_TEACHER_REPR_ON_CPU}'" \
    +actor_rollout_ref.actor.layerwise_opd_vocab_mapping=$LAYERWISE_OPD_VOCAB_MAPPING \
    +actor_rollout_ref.actor.layerwise_opd_jacobian_backward=$LAYERWISE_OPD_JACOBIAN_BACKWARD \
    +actor_rollout_ref.actor.layerwise_opd_student_jlens_checkpoint=$LAYERWISE_OPD_STUDENT_JLENS_CHECKPOINT \
    +actor_rollout_ref.actor.layerwise_opd_gradient_routing=$LAYERWISE_OPD_GRADIENT_ROUTING \
    +actor_rollout_ref.actor.layerwise_opd_readout=$LAYERWISE_OPD_READOUT \
    +actor_rollout_ref.actor.layerwise_opd_detach_lm_head=$LAYERWISE_OPD_DETACH_LM_HEAD \
    +actor_rollout_ref.actor.layerwise_opd_student_affine_checkpoint=$LAYERWISE_OPD_STUDENT_AFFINE_CHECKPOINT \
    +actor_rollout_ref.actor.layerwise_opd_teacher_affine_checkpoint=$LAYERWISE_OPD_TEACHER_AFFINE_CHECKPOINT \
    +actor_rollout_ref.actor.layerwise_opd_diagnostics=$LAYERWISE_OPD_DIAGNOSTICS \
    +actor_rollout_ref.actor.layerwise_opd_diagnostics_interval=$LAYERWISE_OPD_DIAGNOSTICS_INTERVAL \
    +actor_rollout_ref.actor.layerwise_opd_online_jcr_norm_chunk_max_tokens=$LAYERWISE_OPD_ONLINE_JCR_NORM_CHUNK_MAX_TOKENS \
    +actor_rollout_ref.actor.rep_projector_mode=${REP_PROJECTOR_MODE:-full} \
    +actor_rollout_ref.actor.rep_low_rank=${REP_LOW_RANK:-256} \
    ${REP_LOW_RANK_INIT_CHECKPOINT:++actor_rollout_ref.actor.rep_low_rank_init_checkpoint="$REP_LOW_RANK_INIT_CHECKPOINT"} \
    +actor_rollout_ref.actor.rep_ps_projector=${REP_PS_PROJECTOR:-auto} \
    +actor_rollout_ref.actor.rep_mlp_hidden_mult=${REP_MLP_HIDDEN_MULT:-4} \
    +actor_rollout_ref.actor.rep_freeze_ps=${REP_FREEZE_PS:-False} \
    +actor_rollout_ref.actor.rep_head_rank=${REP_HEAD_RANK:-16} \
    ${REP_HEAD_INIT_CHECKPOINT:++actor_rollout_ref.actor.rep_head_init_checkpoint="$REP_HEAD_INIT_CHECKPOINT"} \
    +actor_rollout_ref.actor.use_att_distillation=$USE_ATT_DISTILLATION \
    +actor_rollout_ref.actor.att_distillation_coef=$ATT_DISTILLATION_COEF \
    +actor_rollout_ref.actor.att_distillation_layers=$ATT_DISTILLATION_LAYERS \
    +actor_rollout_ref.actor.att_distillation_positions=$ATT_DISTILLATION_POSITIONS \
    +actor_rollout_ref.actor.att_distillation_last_k=$ATT_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.att_distillation_first_k=$ATT_DISTILLATION_FIRST_K \
    +actor_rollout_ref.actor.att_distillation_max_key_len=$ATT_DISTILLATION_MAX_KEY_LEN \
    +actor_rollout_ref.actor.att_distillation_loss=$ATT_DISTILLATION_LOSS \
    +actor_rollout_ref.actor.att_distillation_temperature=$ATT_DISTILLATION_TEMPERATURE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    ${VLLM_ENFORCE_EAGER:+actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    +reward_model.reward_kwargs.skip_rule_reward_when_rm_scores=$SKIP_RULE_REWARD_WHEN_RM_SCORES \
    +reward_model.reward_kwargs.code_reward_continuous=$CODE_REWARD_CONTINUOUS \
    +reward_model.reward_kwargs.code_reward_all_passed=$CODE_REWARD_ALL_PASSED \
    +reward_model.reward_kwargs.code_reward_lcb_compat=$CODE_REWARD_LCB_COMPAT \
    reward_model.model.path=$REWARD_MODEL_PATH \
    +reward_model.secondary_model_path=$SECONDARY_REWARD_MODEL_PATH \
    +reward_model.tertiary_model_path=$TERTIARY_REWARD_MODEL_PATH \
    reward_model.all_teacher_loss_mean=$ALL_TEACHER_LOSS_MEAN \
    reward_model.expertalign_enabled=$EXPERTALIGN_ENABLED \
    reward_model.teacher_base_model_path=$TEACHER_BASE_MODEL_PATH \
    reward_model.expertalign_weighting=$EXPERTALIGN_WEIGHTING \
    reward_model.expertalign_alignment_epsilon=$EXPERTALIGN_ALIGNMENT_EPSILON \
    "+reward_model.primary_data_sources=$PRIMARY_TEACHER_DATA_SOURCES" \
    "+reward_model.secondary_data_sources=$SECONDARY_TEACHER_DATA_SOURCES" \
    "+reward_model.tertiary_data_sources=$TERTIARY_TEACHER_DATA_SOURCES" \
    +reward_model.model.override_config.attn_implementation=$ATTN_IMPLEMENTATION \
    reward_model.model.trust_remote_code=True \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=$USE_REMOVE_PADDING \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=$REWARD_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.sandbox_fusion.url=$SANDBOX_FUSION_URL \
    reward_model.sandbox_fusion.max_concurrent=$SANDBOX_FUSION_MAX_CONCURRENT \
    reward_model.sandbox_fusion.memory_limit_mb=$SANDBOX_FUSION_MEMORY_LIMIT_MB \
    custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH \
    custom_reward_function.name=$CUSTOM_REWARD_FUNCTION_NAME \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.val_only=$VAL_ONLY \
    trainer.log_val_generations=$TRAINER_LOG_VAL_GENERATIONS \
    "trainer.logger=$TRAINER_LOGGERS" \
    trainer.output_log_path=${PROJECT_PATH}/logs/terminal/${EXPERIMENT_NAME}.log \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=${PROJECT_PATH}/logs/validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=${TOTAL_EPOCHS:-99} \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH \
    trainer.is_plot=$IS_PLOT \
    actor_rollout_ref.actor.checkpoint.load_contents=$ACTOR_CHECKPOINT_LOAD_CONTENTS \
    actor_rollout_ref.actor.checkpoint.save_contents=$ACTOR_CHECKPOINT_SAVE_CONTENTS \
    "${REWARD_BATCH_ARGS[@]}"

# Log the end time for local runs.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
