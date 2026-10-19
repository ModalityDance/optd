#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: launch_multiteacher_math_code_common.sh control|layerwise|raw_layerwise}
case "$MODE" in
  control)
    USE_LAYERWISE=False
    RUN_LABEL=MultiTeacher-MathCode-DeepCoderHarness-OPD-Control-RBS16-150
    ;;
  layerwise)
    USE_LAYERWISE=False
    USE_JOPD=True
    RUN_LABEL=MultiTeacher-MathCode-DeepCoderHarness-JOPD-Single20-Periodic-Last2000-RBS16-150
    ;;
  raw_layerwise)
    USE_LAYERWISE=True
    USE_JOPD=False
    # Default matched to multi-teacher OPD Control (RBS16 x 150). The older
    # RBS16-300 formal arm sets RUN_NAME / TOTAL_STEPS explicitly in its launcher.
    RUN_LABEL=MultiTeacher-MathCode-RawLayerwiseOPDPlusOPD-Legacy-Blocks7-13-19-SelfSupport-Last2000-RBS16-150
    ;;
  repr_mse)
    USE_LAYERWISE=False
    USE_JOPD=False
    RUN_LABEL=MultiTeacher-MathCode-OPRD-MSE-RepOnly-AllLayers-Last2000-RBS16-150
    ;;
  *)
    echo "ERROR: unknown mode $MODE" >&2
    exit 2
    ;;
esac
USE_JOPD=${USE_JOPD:-False}
USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}

DIAGNOSTIC_SMOKE=${DIAGNOSTIC_SMOKE:-False}
STARTUP_SMOKE=${STARTUP_SMOKE:-False}
TOTAL_STEPS=${TOTAL_STEPS:-150}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_VALIDATION_RESPONSE_LENGTH=${MAX_VALIDATION_RESPONSE_LENGTH:-15360}
VALIDATION_RESPONSES=16
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
TEST_INTERVAL=${TEST_INTERVAL:-25}
VAL_BEFORE_TRAIN=False
if [[ "$DIAGNOSTIC_SMOKE" == "True" && "$STARTUP_SMOKE" == "True" ]]; then
  echo "ERROR: DIAGNOSTIC_SMOKE and STARTUP_SMOKE are mutually exclusive" >&2
  exit 2
elif [[ "$STARTUP_SMOKE" == "True" ]]; then
  # This is a mandatory non-formal gate: use the full production validation
  # workload before one real update, then exercise both save and post-update
  # validation at step 1.  It must not shorten token limits or sample counts.
  RUN_LABEL="StartupSmoke-${RUN_LABEL}"
  TOTAL_STEPS=1
  SAVE_INTERVAL=1
  TEST_INTERVAL=1
  VAL_BEFORE_TRAIN=True
  VALIDATION_RESPONSES=1
elif [[ "$STARTUP_SMOKE" != "False" ]]; then
  echo "ERROR: STARTUP_SMOKE must be True or False" >&2
  exit 2
elif [[ "$DIAGNOSTIC_SMOKE" == "True" ]]; then
  RUN_LABEL="DiagnosticSmoke-${RUN_LABEL}"
  TOTAL_STEPS=1
  MAX_RESPONSE_LENGTH=512
  MAX_VALIDATION_RESPONSE_LENGTH=512
  VALIDATION_RESPONSES=1
  SAVE_INTERVAL=-1
  TEST_INTERVAL=-1
elif [[ "$DIAGNOSTIC_SMOKE" != "False" ]]; then
  echo "ERROR: DIAGNOSTIC_SMOKE must be True or False" >&2
  exit 2
fi
# Smoke is a gate, not an experiment: never create a W&B run.
if [[ "$STARTUP_SMOKE" == "True" || "$DIAGNOSTIC_SMOKE" == "True" ]]; then
  TRAINER_LOGGERS='[console]'
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PROJECT_CONDA_ENV=${PROJECT_CONDA_ENV:-optd}
CONDA_EXE=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$CONDA_EXE" && -x "$HOME/data/miniconda3-clean/bin/conda" ]]; then
  CONDA_EXE="$HOME/data/miniconda3-clean/bin/conda"
fi
[[ -x "$CONDA_EXE" ]] || { echo "ERROR: conda executable not found" >&2; exit 1; }

BASE_MODEL=${ACTOR_MODEL_PATH:-/path/to/local/models/DeepSeek-R1-Distill-Qwen-1.5B}
MATH_TEACHER=${REWARD_MODEL_PATH:-/path/to/local/models/JustRL-DeepSeek-1.5B}
CODE_TEACHER=${SECONDARY_REWARD_MODEL_PATH:-/path/to/local/models/DeepCoder-1.5B-Preview-103033da54f104b0d0a3570547fb3721fba99962}
TERTIARY_TEACHER=${TERTIARY_REWARD_MODEL_PATH:-}
if [[ "${EXPERTALIGN_ENABLED:-False}" == "True" ]]; then
  [[ "${LOG_PROB_TOP_K:-16}" == "16" && "${TOP_K_STRATEGY:-only_stu}" == "only_stu" ]] || {
    echo "ERROR: ExpertAlign matched runs require top-16 only_stu support" >&2
    exit 2
  }
  [[ -n "${TEACHER_BASE_MODEL_PATH:-}" && "${TEACHER_BASE_MODEL_PATH}" != "null" && -e "${TEACHER_BASE_MODEL_PATH}" ]] || {
    echo "ERROR: ExpertAlign shared teacher base is missing: ${TEACHER_BASE_MODEL_PATH:-unset}" >&2
    exit 1
  }
fi
if [[ "$USE_JOPD" == "True" ]]; then
  STUDENT_JLENS_CHECKPOINT=${STUDENT_JLENS_CHECKPOINT:?set STUDENT_JLENS_CHECKPOINT for JOPD treatment}
  PRIMARY_TEACHER_JLENS_CHECKPOINT=${PRIMARY_TEACHER_JLENS_CHECKPOINT:?set PRIMARY_TEACHER_JLENS_CHECKPOINT for math teacher}
  SECONDARY_TEACHER_JLENS_CHECKPOINT=${SECONDARY_TEACHER_JLENS_CHECKPOINT:?set SECONDARY_TEACHER_JLENS_CHECKPOINT for code teacher}
fi
TRAIN_DATA=${TRAIN_DATASET:-datasets/multiteacher_math_code_deepcoder_harness_train_150.parquet}
if [[ "${DATA_POOL_MODE:-fixed_2400}" == "full_pool" || "${DATA_POOL_MODE:-fixed_2400}" == "code_teacher_prepared" ]]; then
  [[ "${DATA_SAMPLING_MODE:-}" == "balanced_cycle" && "${SOURCE_HOMOGENEOUS_BATCHES:-False}" == "False" ]] || {
    echo "ERROR: full-pool data requires balanced_cycle mixed batches" >&2
    exit 2
  }
fi
CODE_VAL_DATA=${CODE_VAL_DATA:-datasets/test_data/CodeForces-DeepCoder-Harness/test.parquet}
VALIDATION_FILES=(
  datasets/test_data/AMC23/test.parquet
  datasets/test_data/AIME24/test.parquet
  datasets/test_data/AIME25/test.parquet
  datasets/test_data/AIMO/test.parquet
  "$CODE_VAL_DATA"
)

for path in "$BASE_MODEL" "$MATH_TEACHER" "$CODE_TEACHER" "$TRAIN_DATA" "$CODE_VAL_DATA"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required artifact is missing: $path" >&2
    exit 1
  fi
done
if [[ -n "$TERTIARY_TEACHER" && ! -e "$TERTIARY_TEACHER" ]]; then
  echo "ERROR: tertiary teacher is missing: $TERTIARY_TEACHER" >&2
  exit 1
fi

for path in "${VALIDATION_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: validation dataset is missing: $path" >&2
    exit 1
  fi
done

TRAIN_MANIFEST="${TRAIN_DATA}.manifest.json"
if [[ "$DIAGNOSTIC_SMOKE" == "True" ]]; then
  echo "DIAGNOSTIC_SMOKE: skipping formal 2400-row manifest gate"
elif [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "ERROR: training-data manifest is missing: $TRAIN_MANIFEST" >&2
  exit 1
else
"$CONDA_EXE" run -n "$PROJECT_CONDA_ENV" python -c '
import hashlib, json, pathlib, sys
data_path, manifest_path = map(pathlib.Path, sys.argv[1:3])
manifest = json.loads(manifest_path.read_text())
actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
assert actual == manifest["output_sha256"], "training-data checksum mismatch"
counts = manifest["data_source_counts"]
if sys.argv[3] == "three_task_concat":
    import os
    assert os.environ.get("ALL_TEACHER_LOSS_MEAN", "").lower() == "true"
    assert manifest["pool_mode"] == "three_task_concat"
    assert counts == {"combined_math_code_if": 800} and manifest["rows"] == 800
    assert manifest["source_data_source_counts"] == {
        "math_dapo": 800, "codecontests": 800, "instruction_following": 800,
    }
    assert manifest["reward_semantics"] == "teacher_opd_only_no_rule_reward"
    assert hashlib.sha256(pathlib.Path(manifest["source_file"]).read_bytes()).hexdigest() == manifest["source_sha256"]
    sys.exit(0)
assert sum(counts.values()) == manifest["rows"] and set(counts) in (
    {"math_dapo", "codecontests"},
    {"math_dapo", "codecontests", "instruction_following"},
), counts
pool_mode = sys.argv[3]
if pool_mode == "code_teacher_prepared":
    assert manifest.get("pool_mode") == "code_teacher_prepared"
    assert manifest["rows"] == 35237
    assert counts == {"math_dapo": 17917, "codecontests": 17320}
    assert manifest["code_rows"] == 17320
elif pool_mode == "full_pool":
    assert manifest.get("pool_mode") == "full_pool" and all(counts.values())
    from collections import Counter
    assert Counter(manifest["code_sources"]) == {"taco": 4, "primeintellect": 5, "lcbv5": 11}
else:
    assert manifest["rows"] == 2400 and manifest["steps"] == 150
assert manifest["harness"] == "agentica-project/rllm DeepCoder single-turn prompt contract"
assert manifest["reward_semantics"] in (
    "all_tests_pass_binary_validation_only",
    "teacher_opd_only_no_rule_reward",
)
assert manifest["deepcoder_dataset_revision"] == "177913a7bd43791646ef6a43645caa3c871ab3db"
' "$TRAIN_DATA" "$TRAIN_MANIFEST" "${DATA_POOL_MODE:-fixed_2400}"
fi

if [[ "$DIAGNOSTIC_SMOKE" != "True" && ( -z "${SANDBOX_FUSION_URL:-}" || "${SANDBOX_FUSION_URL}" == "null" ) ]]; then
  echo "ERROR: SANDBOX_FUSION_URL is required; untrusted code must not run on the host" >&2
  exit 1
fi
# SandboxFusion's service route is /run_code.  An older A800 tunnel note used
# /v1/run_code, which answers the health check but returns 404 for every real
# code-evaluation POST.  Normalize that obsolete spelling before any worker is
# started; the preflight below then executes the same POST contract as reward.
if [[ "$SANDBOX_FUSION_URL" == */v1/run_code ]]; then
  SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL%/v1/run_code}/run_code"
  echo "Normalized obsolete SandboxFusion endpoint to: $SANDBOX_FUSION_URL"
fi

EXPECTED_SHARD_1=db6993b17d3fb7ba01052738e99ca44e8515eeec2a7952a7aaaff41885f4d363
EXPECTED_SHARD_2=ef94976d8f1fa3e9bb374e2f156a97c7d51fe6f92572e46670bc4caf6b9317ec
if [[ "${MODEL_FAMILY:-deepseek}" == "qwen3_4b" ]]; then
  "$CONDA_EXE" run -n "$PROJECT_CONDA_ENV" python scripts/preflight_qwen4b_teachers.py "$BASE_MODEL" "$MATH_TEACHER" "$CODE_TEACHER"
elif [[ "${MODEL_FAMILY:-deepseek}" == "r1_distill_7b" ]]; then
  "$CONDA_EXE" run -n "$PROJECT_CONDA_ENV" python scripts/preflight_r1distill7b_teachers.py "$BASE_MODEL" "$MATH_TEACHER" "$CODE_TEACHER"
elif [[ "$DIAGNOSTIC_SMOKE" != "True" ]]; then printf '%s  %s\n%s  %s\n' \
  "$EXPECTED_SHARD_1" "$CODE_TEACHER/model-00001-of-00002.safetensors" \
  "$EXPECTED_SHARD_2" "$CODE_TEACHER/model-00002-of-00002.safetensors" | sha256sum --check --status || {
    echo "ERROR: DeepCoder weight checksum mismatch" >&2
    exit 1
  }; fi

"$CONDA_EXE" run -n "$PROJECT_CONDA_ENV" python -c '
import json, sys
base, code = (json.load(open(path)) for path in sys.argv[1:])
base_merges = [tuple(item.split(" ", 1) if isinstance(item, str) else item) for item in base["model"]["merges"]]
code_merges = [tuple(item.split(" ", 1) if isinstance(item, str) else item) for item in code["model"]["merges"]]
assert base["model"]["vocab"] == code["model"]["vocab"], "teacher vocab ids differ"
assert base_merges == code_merges, "teacher BPE merges differ"
' "$BASE_MODEL/tokenizer.json" "$CODE_TEACHER/tokenizer.json"

# The A800 jlens environment has an editable verl install.  Pin imports to
# this experiment checkout and fail before allocating GPUs if either the
# dispatch table or the validation parquet sources are inconsistent.
RUN_PYTHONPATH="$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
PREFLIGHT_ARGS=()
for path in "${VALIDATION_FILES[@]}"; do
  PREFLIGHT_ARGS+=(--validation-file "$path")
done
if [[ "$DIAGNOSTIC_SMOKE" != "True" ]]; then PYTHONPATH="$RUN_PYTHONPATH" "$CONDA_EXE" run -n "$PROJECT_CONDA_ENV" python \
  scripts/preflight_multiteacher_validation.py \
  --expected-root "$REPO_ROOT" \
  --sandbox-url "$SANDBOX_FUSION_URL" \
  "${PREFLIGHT_ARGS[@]}"; fi

TRAIN_GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -r -a GPUS <<< "$TRAIN_GPUS"
if [[ "${#GPUS[@]}" -ne 4 && !( "${MODEL_FAMILY:-}" == "r1_distill_7b" && "${#GPUS[@]}" -eq 8 ) ]]; then
  echo "ERROR: formal multi-teacher run requires four GPUs (R1-Distill-7B also supports eight)" >&2
  exit 1
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "ERROR: non-numeric GPU id $gpu" >&2; exit 1; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "ERROR: duplicate GPU id $gpu" >&2; exit 1; }
  SEEN_GPUS[$gpu]=1
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "ERROR: GPU $gpu has a compute process; re-check ownership" >&2
    exit 1
  fi
done

RUN_NAME=${RUN_NAME:-token_reward_direct_DAPO-DeepCoder-Mixed_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepCoder-1.5B_rbs16-pmbs8-n2-topk16-2026-08-29-${RUN_LABEL}}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/multiteacher_math_code}
RAY_PORT=${RAY_PORT:-6384}
RAY_START=${RAY_START:-True}
mkdir -p "$RAY_TEMP_DIR"

# The 32k actor token cap OOMs during backward for this 16k-response run.
# Keep the validated 18432 cap as the launcher default.

exec env \
  PYTHONPATH="$RUN_PYTHONPATH" \
  OPRD_EXPECTED_REPO_ROOT="$REPO_ROOT" \
  MULTITEACHER_PREFLIGHT_SCRIPT="${MULTITEACHER_PREFLIGHT_SCRIPT-$REPO_ROOT/scripts/preflight_multiteacher_validation.py}" \
  MULTITEACHER_VALIDATION_FILES="${VALIDATION_FILES[*]}" \
  MULTITEACHER_SANDBOX_URL="$SANDBOX_FUSION_URL" \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  ACTOR_MODEL_PATH="$BASE_MODEL" \
  REWARD_MODEL_PATH="$MATH_TEACHER" \
  SECONDARY_REWARD_MODEL_PATH="$CODE_TEACHER" \
  TERTIARY_REWARD_MODEL_PATH="${TERTIARY_TEACHER:-null}" \
  PRIMARY_TEACHER_DATA_SOURCES="${PRIMARY_TEACHER_DATA_SOURCES:-[math_dapo]}" \
  SECONDARY_TEACHER_DATA_SOURCES="${SECONDARY_TEACHER_DATA_SOURCES:-[codecontests]}" \
  TERTIARY_TEACHER_DATA_SOURCES="${TERTIARY_TEACHER_DATA_SOURCES:-[]}" \
  NO_PREASSISTANT_EOS_DATA_SOURCES="${NO_PREASSISTANT_EOS_DATA_SOURCES:-[]}" \
  TRAIN_DATASET="$TRAIN_DATA" \
  TRAIN_DATASET_NAME=DAPO-DeepCoder-Mixed-8x8 \
  SOURCE_HOMOGENEOUS_BATCHES=${SOURCE_HOMOGENEOUS_BATCHES:-False} \
  DATA_SAMPLING_MODE="${DATA_SAMPLING_MODE:-}" \
  TEST_DATASET='[datasets/test_data/AMC23/test.parquet,datasets/test_data/AIME24/test.parquet,datasets/test_data/AIME25/test.parquet,datasets/test_data/AIMO/test.parquet,datasets/test_data/CodeForces-DeepCoder-Harness/test.parquet]' \
  CUSTOM_REWARD_FUNCTION_PATH=null \
  CUSTOM_REWARD_FUNCTION_NAME=reward_func \
  SKIP_RULE_REWARD_WHEN_RM_SCORES=True \
  CODE_REWARD_CONTINUOUS=False \
  CODE_REWARD_ALL_PASSED=True \
  CODE_REWARD_LCB_COMPAT=True \
  CHAT_TEMPLATE_PROFILE="${CHAT_TEMPLATE_PROFILE:-rllm_deepseek_qwen}" \
  SANDBOX_FUSION_URL="$SANDBOX_FUSION_URL" \
  SANDBOX_FUSION_MAX_CONCURRENT=${SANDBOX_FUSION_MAX_CONCURRENT:-32} \
  SANDBOX_FUSION_MEMORY_LIMIT_MB=${SANDBOX_FUSION_MEMORY_LIMIT_MB:-1024} \
  N_GPUS_PER_NODE="${#GPUS[@]}" \
  PARALLEL_SIZE=1 \
  TOTAL_TRAINING_STEPS="$TOTAL_STEPS" \
  ROLLOUT_PROMPT_BATCH_SIZE=${ROLLOUT_PROMPT_BATCH_SIZE:-16} \
  PPO_PROMPT_MINI_BATCH_SIZE=${PPO_PROMPT_MINI_BATCH_SIZE:-8} \
  MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8} \
  PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
  REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1} \
  N_RESPONSES="${ROLLOUT_N:-2}" \
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048} \
  MAX_RESP_LENGTH="$MAX_RESPONSE_LENGTH" \
  VAL_N_RESPONSES="$VALIDATION_RESPONSES" \
  MAX_VAL_RESP_LENGTH="$MAX_VALIDATION_RESPONSE_LENGTH" \
  LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16} \
  TOP_K_STRATEGY=only_stu \
  REWARD_WEIGHT_MODE=student_p \
  USE_LAYERWISE_OPD_DISTILLATION="$USE_LAYERWISE" \
  USE_JLENS_DISTILLATION="$USE_JOPD" \
  USE_REP_DISTILLATION="$USE_REP_DISTILLATION" \
  REP_DISTILLATION_ONLY="$REP_DISTILLATION_ONLY" \
  REP_DISTILLATION_COEF="${REP_DISTILLATION_COEF:-1.0}" \
  REP_DISTILLATION_CROSSFADE_STEPS="${REP_DISTILLATION_CROSSFADE_STEPS:-0}" \
  REP_DISTILLATION_POSITIONS="${REP_DISTILLATION_POSITIONS:-last_k}" \
  REP_DISTILLATION_LAST_K="${REP_DISTILLATION_LAST_K:-2000}" \
  REP_DISTILLATION_FIRST_K="${REP_DISTILLATION_FIRST_K:-2000}" \
  REP_DISTILLATION_LAYERS="${REP_DISTILLATION_LAYERS:-all}" \
  STUDENT_JLENS_CHECKPOINT="${STUDENT_JLENS_CHECKPOINT:-null}" \
  TEACHER_JLENS_CHECKPOINT="${PRIMARY_TEACHER_JLENS_CHECKPOINT:-null}" \
  PRIMARY_TEACHER_JLENS_CHECKPOINT="${PRIMARY_TEACHER_JLENS_CHECKPOINT:-null}" \
  SECONDARY_TEACHER_JLENS_CHECKPOINT="${SECONDARY_TEACHER_JLENS_CHECKPOINT:-null}" \
  JALIGN_LOSS_TYPE=opd \
  JALIGN_STUDENT_TRANSPORT=periodic_refresh \
  JALIGN_GRADIENT_ROUTING=jacobian \
  JALIGN_OPD_SUPPORT=final_policy \
  JALIGN_OPD_AGGREGATION=legacy_additive \
  JALIGN_COEF=1.0 \
  JALIGN_STUDENT_LAYERS=20 \
  JALIGN_TEACHER_LAYERS=20 \
  JALIGN_POSITIONS=last_k \
  JALIGN_LAST_K=2000 \
  JALIGN_REFRESH_INTERVAL=25 \
  JALIGN_REFRESH_CUDA_VISIBLE_DEVICES="${JALIGN_REFRESH_CUDA_VISIBLE_DEVICES:-7}" \
  ACTOR_CHECKPOINT_SAVE_CONTENTS='[model,optimizer,extra,hf_model]' \
  LAYERWISE_OPD_SUPPORT="${LAYERWISE_OPD_SUPPORT:-self_student}" \
  LAYERWISE_OPD_AGGREGATION="${LAYERWISE_OPD_AGGREGATION:-equal_weight}" \
  LAYERWISE_OPD_COEF="${LAYERWISE_OPD_COEF:-1.0}" \
  LAYERWISE_OPD_TEMPERATURE="${LAYERWISE_OPD_TEMPERATURE:-1.0}" \
  LAYERWISE_OPD_TOKEN_CHUNK_SIZE="${LAYERWISE_OPD_TOKEN_CHUNK_SIZE:-32}" \
  LAYERWISE_OPD_OBJECTIVE="${LAYERWISE_OPD_OBJECTIVE:-absolute}" \
  LAYERWISE_OPD_TRANSITION_WEIGHTS="${LAYERWISE_OPD_TRANSITION_WEIGHTS:-}" \
  LAYERWISE_OPD_TOP_K="${LAYERWISE_OPD_TOP_K:-16}" \
  LAYERWISE_OPD_POSITIONS="${LAYERWISE_OPD_POSITIONS:-last_k}" \
  LAYERWISE_OPD_LAST_K="${LAYERWISE_OPD_LAST_K:-2000}" \
  LAYERWISE_OPD_STUDENT_LAYERS="${LAYERWISE_OPD_STUDENT_LAYERS:-7,13,19}" \
  LAYERWISE_OPD_TEACHER_LAYERS="${LAYERWISE_OPD_TEACHER_LAYERS:-7,13,19}" \
  LAYERWISE_OPD_GRADIENT_ROUTING="${LAYERWISE_OPD_GRADIENT_ROUTING:-raw}" \
  LAYERWISE_OPD_READOUT="${LAYERWISE_OPD_READOUT:-raw}" \
  LAYERWISE_OPD_DETACH_LM_HEAD="${LAYERWISE_OPD_DETACH_LM_HEAD:-False}" \
  LR_SCHEDULER="${LR_SCHEDULER:-constant}" \
  SAVE_FREQ="$SAVE_INTERVAL" \
  TEST_FREQ="$TEST_INTERVAL" \
  TRAINER_LOGGERS="${TRAINER_LOGGERS:-[console,wandb]}" \
  TRAINER_LOG_VAL_GENERATIONS=0 \
  VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
  RESUME_MODE=${RESUME_MODE:-auto} \
  IS_PLOT=False \
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}" \
  USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}" \
  HF_HUB_OFFLINE=1 \
  FREE_CACHE_ENGINE=True \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" \
  DATA_SAMPLER_CLASS_PATH="${DATA_SAMPLER_CLASS_PATH:-}" \
  DATA_SAMPLER_CLASS_NAME="${DATA_SAMPLER_CLASS_NAME:-}" \
  MODEL_DTYPE=bfloat16 \
  ENTROPY_FROM_LOGITS_WITH_CHUNKING=True \
  LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}" \
  PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-18432} \
  LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 \
  USE_TORCH_COMPILE_OVERRIDE="${USE_TORCH_COMPILE_OVERRIDE:-}" \
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}" \
  RUN_LABEL="$RUN_LABEL" \
  CKPT_PATH="outputs/$RUN_NAME" \
  EXPERIMENT_NAME="$RUN_NAME" \
  LOG_FILE="${LOG_FILE_OVERRIDE:-logs/${RUN_NAME}.launch.log}" \
  RAY_STOP_BEFORE_RUN=False \
  RAY_START_BEFORE_RUN=$RAY_START \
  RAY_ADDRESS=127.0.0.1:$RAY_PORT \
  RAY_START_ARGS="--port=$RAY_PORT --temp-dir=$RAY_TEMP_DIR --include-dashboard=False --num-cpus=32 --num-gpus=${#GPUS[@]}" \
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-99}" \
  "$CONDA_EXE" run --no-capture-output -n "$PROJECT_CONDA_ENV" bash on_policy_distillation.sh
