#!/bin/bash
# Shared Ray bootstrap helpers for Narval multi-node PAG jobs.
# Source from smoke/train slurm scripts after activate_pag.sh.
#
# Narval pitfalls encoded here:
# - hostname supports -i, NOT GNU -I
# - AF_UNIX socket path must be <=107 bytes → use /tmp/r$SLURM_JOB_ID (not SLURM_TMPDIR)

ray_resolve_head_ip() {
  local head_node="$1"
  local ip
  ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i 2>/dev/null | awk '{print $1}')
  if [[ -z "${ip}" || "${ip}" == "127.0.0.1" ]]; then
    ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
      bash -c "getent hosts \$(hostname) | awk '{print \$1; exit}'")
  fi
  if [[ -z "${ip}" || "${ip}" == "127.0.0.1" ]]; then
    echo "[ray_bootstrap] ERROR: failed to resolve IP for ${head_node}" >&2
    return 1
  fi
  # crude IPv4 check
  if [[ ! "${ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "[ray_bootstrap] ERROR: malformed IP '${ip}' for ${head_node}" >&2
    return 1
  fi
  echo "$ip"
}

ray_env() {
  source /scratch/yuranli/virtualenvs/PAG/bin/activate_pag.sh
  export PYTHONNOUSERSITE=1
  export VLLM_USE_V1=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export TOKENIZERS_PARALLELISM=true
  # MUST stay short (Ray plasma AF_UNIX <=107)
  export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
  export TMPDIR="$RAY_TMPDIR"
  mkdir -p "$RAY_TMPDIR"

  # Worst-case socket path length check before ray start
  local probe="${RAY_TMPDIR}/ray/session_YYYY-MM-DD_HH-MM-SS_mmmmmm_PPPPPP/sockets/plasma_store"
  if (( ${#probe} > 107 )); then
    echo "[ray_bootstrap] ERROR: RAY_TMPDIR too long for AF_UNIX (${#probe}>107): ${RAY_TMPDIR}" >&2
    return 1
  fi
  echo "[ray_bootstrap] node=$(hostname) ip_hint=$(hostname -i 2>/dev/null | awk '{print $1}') RAY_TMPDIR=${RAY_TMPDIR} socket_probe_len=${#probe}"
}

ray_start_head() {
  local head_ip="$1"
  local port="$2"
  local n_gpus="$3"
  set -euo pipefail
  ray_env
  ray stop --force >/dev/null 2>&1 || true
  echo "[ray_bootstrap] starting HEAD node-ip=${head_ip} port=${port} gpus=${n_gpus} temp=${RAY_TMPDIR}"
  ray start --head --node-ip-address="${head_ip}" --port="${port}" \
    --temp-dir="${RAY_TMPDIR}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${n_gpus}" \
    --dashboard-host=127.0.0.1
  echo "[ray_bootstrap] HEAD started"
}

ray_start_worker() {
  local head_ip="$1"
  local port="$2"
  local n_gpus="$3"
  set -euo pipefail
  ray_env
  ray stop --force >/dev/null 2>&1 || true
  echo "[ray_bootstrap] starting WORKER address=${head_ip}:${port} gpus=${n_gpus} temp=${RAY_TMPDIR}"
  ray start --address="${head_ip}:${port}" \
    --temp-dir="${RAY_TMPDIR}" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${n_gpus}"
  echo "[ray_bootstrap] WORKER started"
}

ray_wait_ready() {
  local address="$1"
  local tries="${2:-60}"
  local i
  for i in $(seq 1 "$tries"); do
    if ray status --address="$address" >/dev/null 2>&1; then
      echo "[ray_bootstrap] ray ready at ${address} (try ${i})"
      ray status --address="$address" || true
      return 0
    fi
    sleep 2
  done
  echo "[ray_bootstrap] ERROR: ray not ready at ${address} after ${tries} tries" >&2
  return 1
}
