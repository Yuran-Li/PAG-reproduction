#!/bin/bash
#SBATCH --job-name=pag-train-nibi
#SBATCH --account=def-bboulet_gpu
#SBATCH --partition=gpubase_bynode_b5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Nibi full train: 1×8 H100, ~40–50h estimated (7B Instruct, 40 epochs).
# Single-node Ray — do NOT wrap this script in srun (exclusive bynode).
#
#   cd /scratch/yuranli/verify_then_rectify/PAG-reproduction
#   mkdir -p logs
#   sbatch quick_start/train_pag_nibi.slurm
#
# Optional overrides:
#   sbatch --export=ALL,USE_WANDB=1,EXPERIMENT_NAME=my_run,MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
#     quick_start/train_pag_nibi.slurm
#
# Smoke instead: use quick_start/smoke_pag_nibi.slurm (or --time=1:00:00
# --partition=gpubase_bynode_b2 --export=ALL,SMOKE=1 on this file).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
if [[ "$(basename "$REPO_ROOT")" == "quick_start" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
# shellcheck source=/dev/null
source "$REPO_ROOT/quick_start/ray_bootstrap_narval.sh"

export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/checkpoints}"
# Paper base models: Qwen2.5-*-Instruct (NOT Math-*). Default: 7B Instruct.
# run_pag_local.sh picks train data: 7B → dapo17k.parquet, else math7500.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
export N_GPUS=8
export NNODES=1
export SMOKE="${SMOKE:-0}"
export USE_WANDB="${USE_WANDB:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_pag_8xh100_nibi}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
# Instruct models support long context; keep paper-script lengths (1024/2048).
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
# Optional hard override: TRAIN_FILE=/path/to.parquet

# Single-node exclusive bynode: the batch script already runs on the allocated
# node. Do NOT call srun here — a second step has no leftover GPUs/CPUs and
# fails with: "Requested node configuration is not available".
head_node="$(hostname)"
port=6379
echo "[train-nibi] resolving head IP locally (no srun)..."
head_ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
if [[ -z "${head_ip}" || "${head_ip}" == "127.0.0.1" ]]; then
  head_ip="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
fi
if [[ -z "${head_ip}" || "${head_ip}" == "127.0.0.1" || ! "${head_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "[train-nibi] ERROR: failed to resolve local IP for ${head_node} (got '${head_ip:-}')" >&2
  exit 1
fi
echo "[train-nibi] head=${head_node} ip=${head_ip}"
echo "[train-nibi] MODEL_PATH=${MODEL_PATH} N_GPUS=${N_GPUS} NNODES=${NNODES} GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[train-nibi] MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "[train-nibi] SMOKE=${SMOKE} EXPERIMENT_NAME=${EXPERIMENT_NAME} USE_WANDB=${USE_WANDB}"
echo "[train-nibi] RAY_TMPDIR will be /tmp/r${SLURM_JOB_ID}"

# Single-node: start Ray head in this batch step (no srun)
ray_start_head "${head_ip}" "${port}" "${N_GPUS}"
export RAY_ADDRESS="${head_ip}:${port}"
ray_wait_ready "$RAY_ADDRESS" 60

cleanup() {
  echo '[train-nibi] shutting down ray...'
  ray stop --force || true
}
trap cleanup EXIT

echo "[train-nibi] RAY_ADDRESS=${RAY_ADDRESS}"
ray status --address="$RAY_ADDRESS" || true

set +e
env RAY_ADDRESS="$RAY_ADDRESS" \
    N_GPUS="$N_GPUS" \
    NNODES="$NNODES" \
    SMOKE="$SMOKE" \
    MODEL_PATH="$MODEL_PATH" \
    CKPT_PATH="$CKPT_PATH" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    USE_WANDB="$USE_WANDB" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
    ${TRAIN_FILE:+TRAIN_FILE="$TRAIN_FILE"} \
    bash "$REPO_ROOT/quick_start/run_pag_local.sh"
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "[train-nibi] PASS (exit=0)."
else
  echo "[train-nibi] FAIL (exit=${status}). Check logs for CUDA OOM / Ray / vLLM errors."
fi
exit "$status"
