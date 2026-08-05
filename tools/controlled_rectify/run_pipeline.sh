#!/usr/bin/env bash
# PAG controlled rectification pipeline (GPT critiques + Pre-RL/PPO eval).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CR="$ROOT/tools/controlled_rectify"
PY="${PAG_PY:-/data/yuranli/envs/PAG/bin/python}"
S2R_EVAL_PY="${S2R_EVAL_PY:-/data/yuranli/envs/S2R_eval/bin/python}"

MATH500="${MATH500:-$ROOT/datasets/math500.parquet}"
PRERL="${PRERL:-/data/yuranli/hf-cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
PPO_HF="${PPO_HF:-$ROOT/checkpoints/PAG/qwen1p5b_pag/global_step_400/actor_hf}"
PPO_ACTOR="${PPO_ACTOR:-$ROOT/checkpoints/PAG/qwen1p5b_pag/global_step_400/actor}"

POOL="${POOL:-$CR/data/fixed_wrong_pag_prerl.jsonl}"
DATA="${DATA:-$CR/data/fixed_wrong_pag_prerl_with_critique.jsonl}"
RESULTS="${RESULTS:-$CR/results}"
mkdir -p "$CR/data" "$RESULTS"

STAGE="${1:-all}"
TP="${TP:-2}"
CRITIC_MODEL="${GPT_CRITIQUE_MODEL:-gpt-5o}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Match PAG train/eval: vLLM V1 profiler OOMs on this stack
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "[$(date)] stage=$STAGE ROOT=$ROOT"
echo "[$(date)] VLLM_USE_V1=$VLLM_USE_V1 GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_NUM_SEQS=$MAX_NUM_SEQS TP=$TP"

merge_ppo() {
  if [[ -f "$PPO_HF/config.json" ]]; then
    echo "[$(date)] PPO HF already exists: $PPO_HF"
    return
  fi
  echo "[$(date)] Merging FSDP actor -> $PPO_HF"
  "$PY" "$ROOT/scripts/model_merger.py" \
    --backend fsdp \
    --hf_model_path "$PRERL" \
    --local_dir "$PPO_ACTOR" \
    --target_dir "$PPO_HF"
}

pool() {
  echo "[$(date)] Build Pre-RL oracle wrong pool"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PY" "$CR/build_prerl_pool.py" \
    --model_path "$PRERL" \
    --parquet "$MATH500" \
    --out "$POOL" \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_num_seqs "$MAX_NUM_SEQS"
}

critique_gpt() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: export OPENAI_API_KEY=... (optional OPENAI_BASE_URL / GPT_CRITIQUE_MODEL)"
    exit 1
  fi
  echo "[$(date)] GPT fixed critiques model=$CRITIC_MODEL"
  # openai client: prefer S2R_eval or PAG env
  local RUN_PY="$PY"
  if ! "$PY" -c "import openai" 2>/dev/null; then
    RUN_PY="$S2R_EVAL_PY"
  fi
  "$RUN_PY" "$CR/generate_fixed_critiques_gpt.py" \
    --input "$POOL" \
    --output "$DATA" \
    --model "$CRITIC_MODEL" \
    --skip_existing
}

eval_one() {
  local name="$1"
  local model="$2"
  local n="$3"
  local temp="$4"
  local tag="$5"
  echo "[$(date)] eval $name $tag n=$n T=$temp"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$PY" "$CR/controlled_rectify_eval.py" \
    --model_path "$model" \
    --data "$DATA" \
    --conditions gen fix regen \
    --n_samples "$n" --temperature "$temp" \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --output "$RESULTS/${name}_${tag}.jsonl"
}

eval_at1() {
  eval_one pag_prerl "$PRERL" 1 0.0 at1
  eval_one pag_ppo400 "$PPO_HF" 1 0.0 at1
}

eval_at8() {
  eval_one pag_prerl "$PRERL" 8 0.7 at8
  eval_one pag_ppo400 "$PPO_HF" 8 0.7 at8
}

aggregate() {
  "$PY" "$CR/aggregate_table.py" --results_dir "$RESULTS"
}

case "$STAGE" in
  merge) merge_ppo ;;
  pool) pool ;;
  critique|critique_gpt) critique_gpt ;;
  eval_at1) eval_at1 ;;
  eval_at8) eval_at8 ;;
  aggregate) aggregate ;;
  all)
    merge_ppo
    pool
    critique_gpt
    eval_at1
    eval_at8
    aggregate
    ;;
  *)
    echo "Usage: $0 {merge|pool|critique|eval_at1|eval_at8|aggregate|all}"
    exit 1
    ;;
esac

echo "[$(date)] DONE stage=$STAGE"
