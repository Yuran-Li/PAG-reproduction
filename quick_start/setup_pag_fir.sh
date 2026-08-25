#!/usr/bin/env bash
# Create the PAG venv on Fir (same path/modules as Nibi). Do NOT rsync the venv.
# Run on a Fir login node:
#   bash quick_start/setup_pag_fir.sh
set -euo pipefail

VENV="${VENV:-/scratch/yuranli/virtualenvs/PAG}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10.13
mkdir -p "$(dirname "$VENV")"
if [[ ! -x "$VENV/bin/python" ]]; then
  virtualenv --no-download "$VENV"
fi

cat > "$VENV/bin/activate_pag.sh" <<'EOF'
#!/bin/bash
if ! module is-loaded StdEnv/2023 2>/dev/null; then module load StdEnv/2023; fi
if ! module is-loaded gcc/12.3 2>/dev/null; then module load gcc/12.3; fi
if ! module is-loaded cuda/12.2 2>/dev/null; then module load cuda/12.2; fi
if ! module is-loaded python/3.10.13 2>/dev/null; then module load python/3.10.13; fi
# shellcheck source=/dev/null
source /scratch/yuranli/virtualenvs/PAG/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HF_HOME="${HF_HOME:-/scratch/yuranli/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
echo "[activate_pag] $(hostname) python=$(command -v python) $(python -V 2>&1) VIRTUAL_ENV=${VIRTUAL_ENV}"
EOF

# shellcheck source=/dev/null
source "$VENV/bin/activate_pag.sh"
export PYTHONNOUSERSITE=1
pip install --no-index --upgrade pip setuptools wheel
pip install --no-index 'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0' || pip install --no-index torch torchvision torchaudio
pip install --no-index 'vllm==0.8.4' || pip install --no-index vllm
pip install --no-index flash-attn || true
pip install --no-index 'numpy==1.26.4' 'accelerate==1.4.0' 'transformers==4.51.3' || true
cd "$REPO_ROOT"
pip install --no-index -e . || pip install -e .
pip install --no-index 'math-verify==0.9.0' || pip install 'math-verify==0.9.0'
python - <<'PY'
import torch, vllm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)
print('cuda_available', torch.cuda.is_available())
PY
echo "[setup-fir] venv ready at ${VENV}"
echo "[setup-fir] next: sbatch --time=1:00:00 --partition=gpubase_bynode_b1 --export=ALL,SMOKE=1,EXPERIMENT_NAME=smoke_qwen7b_fir quick_start/train_pag_fir.slurm"
