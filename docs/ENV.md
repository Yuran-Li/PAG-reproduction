# Environment setup (local reproduction)

This fork trains/evals PAG with **FSDP + vLLM**. Match the critical stack below; a full `pip freeze` alone often fails across machines because of CUDA wheels.

## Dependency file roles (no conflict)

| File | Role | Use when |
|------|------|----------|
| [`requirements.txt`](../requirements.txt) | Declared install entry (loose) | After torch/vllm/flash-attn are in place, or as a checklist |
| [`environment.yml`](../environment.yml) | Conda skeleton + critical pins | Creating a fresh `PAG` env |
| [`requirements.freeze.txt`](../requirements.freeze.txt) | Snapshot of a **known-good** machine | Diff / debug only — **do not** `pip install -r` blindly |
| [`docker/Dockerfile.ngc.vllm0.8`](../docker/Dockerfile.ngc.vllm0.8) | Closest upstream container recipe | Prefer if you can run NGC images |

Default install path: **ENV.md steps** → `pip install -e .` → `math-verify` if missing.

## Critical versions (known-good, 2026-07-27)

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| CUDA (torch build) | 12.4 |
| `torch` | 2.6.0 (`cxx11_abi=False`) |
| `vllm` | 0.8.2 |
| `flash-attn` | 2.7.4.post1 (`+cu12torch2.6cxx11abiFALSE`, cp310) |
| `transformers` | 4.51.3 |
| `accelerate` | 1.4.0 |
| `numpy` | 1.26.4 |
| `math-verify` | 0.9.0 |
| `tensordict` | ≤0.6.2 (we used 0.6.2) |

Upstream verl base: commit **81a15ed7** (see original README).

## Recommended local install

```bash
# 1) Env
conda create -n PAG python=3.10 -y
conda activate PAG
export PYTHONNOUSERSITE=1   # avoid ~/.local pollution

# 2) Torch + vLLM (order matters)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install vllm==0.8.2

# 3) FlashAttention wheel (do not pip build from source unless you must)
WHL=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget -nv -O "$WHL" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$WHL"
pip install "$WHL"

# 4) This repo + math verify
cd /path/to/Policy-As-GenVerifier
pip install -e .
pip install math-verify==0.9.0
# optional: remaining declared deps
pip install -r requirements.txt

# 5) Runtime flags used in our runs
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
```

Optional conda shortcut (still finish flash-attn + `pip install -e .` manually):

```bash
conda env create -f environment.yml
conda activate PAG
# then flash-attn wheel + pip install -e . as above
```

## Local launch scripts

- Train: `quick_start/run_pag_local.sh` (override `MODEL_PATH`, `CKPT_PATH`, `N_GPUS`, `USE_WANDB=1`)
- Eval: `quick_start/run_eval_local.sh` (`RESUME_PATH`, `VAL_N`, `REVISE_GATE=pag|oracle|always`)

Set data/model roots via env rather than hard-coding:

```bash
export HF_HOME=/path/to/hf-cache
export MODEL_PATH=...   # Qwen2.5-1.5B-Instruct snapshot
export CKPT_PATH=...    # checkpoints root
```

## Known pitfalls

1. **`VLLM_USE_V1=0`** — vLLM 0.8 V1 engine can fail dtype profiling on this stack; we force V0.
2. **`PYTHONNOUSERSITE=1`** — user-site packages can break `accelerate` / numpy detection.
3. **`accelerate` + `numpy`** — pin `accelerate==1.4.0` and `numpy==1.26.4` (numpy 2.x / broken `_core` stubs caused import failures).
4. **FlashAttention ABI** — must match torch `cxx11abiFALSE` wheel; wrong wheel → import or runtime crash.
5. **8-GPU FSDP checkpoints** — loading `global_step_*` sharded ckpts expects the same world size (typically 8).
6. Do **not** install this `verl` into an unrelated env (e.g. S2R); keep a dedicated `PAG` env.

## Refreshing the freeze snapshot

On a machine that already trains successfully:

```bash
conda activate PAG
pip freeze > requirements.freeze.txt
# Then manually:
#   - remove editable `-e git+...#egg=verl` → comment `# pip install -e .`
#   - replace `flash_attn @ file://...` with the wheel URL comment (see file header)
```

## What not to commit

- `checkpoints/`, `wandb/`, large datasets, HF weights  
- API tokens / `.netrc`  
- Machine-absolute paths inside scripts (prefer env vars)
