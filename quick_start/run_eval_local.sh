#!/usr/bin/env bash
# Formal PAG eval (paper-style): val_only, dump trajectories for rectify-gap analysis.
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/data/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/yuranli/hf-cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yuranli/hf-cache/hub}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONNOUSERSITE=1

math500="$REPO_ROOT/datasets/math500.parquet"
aime2024="$REPO_ROOT/datasets/aime2024.parquet"
aime2025="$REPO_ROOT/datasets/aime2025.parquet"
minervamath="$REPO_ROOT/datasets/minervamath.parquet"
dapo17k="$REPO_ROOT/datasets/dapo17k.parquet"

PROJECT_NAME='PAG'
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
MODEL_PATH="${MODEL_PATH:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
RESUME_PATH="${RESUME_PATH:-$CKPT_PATH/PAG/qwen1p5b_pag/global_step_400}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_pag_qwen1p5b_step400_dump}"
N_GPUS="${N_GPUS:-8}"
# val_only + grpo (no critic): can use higher util than training
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
VAL_N="${VAL_N:-32}"
# pag | oracle | always
REVISE_GATE="${REVISE_GATE:-pag}"

n=4
rollout_type=pag
num_turns=2

# Default: MATH-500 only (for rectify-gap dump). Override VAL_FILES for full suite.
VAL_FILES="${VAL_FILES:-['$math500']}"
VALIDATION_JSON="${VALIDATION_JSON:-$REPO_ROOT/validation_results/pag_step400_math500_${REVISE_GATE}.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}_${REVISE_GATE}"

python3 -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_PATH" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    algorithm.adv_estimator=grpo \
    data.train_files="[$dapo17k]" \
    data.val_files="$VAL_FILES" \
    data.filter_overlong_prompts=True \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2028 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.num_turns=$num_turns \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.rollout_type=$rollout_type \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.num_turns=$num_turns \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.revise_gate=$REVISE_GATE \
    reward_model.policy_rs=True \
    reward_model.rs_coef=1.0 \
    algorithm.norm_type=role \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.log_val_generations=2 \
    trainer.save_validation_results=True \
    trainer.validation_results_path="$VALIDATION_JSON"
