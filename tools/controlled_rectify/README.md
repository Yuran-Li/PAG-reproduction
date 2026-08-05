# Controlled Rectification Eval (PAG)

Fill the controlled-rectification table for **PAG Pre-RL** vs **PPO (global_step_400)** on MATH-500.

| Column | Meaning |
|--------|---------|
| \(\mathrm{ECR}_{\mathrm{gen}}\) | PAG turns: \(y_0\) + generic verify (“… The answer is wrong.”) + regenerate user |
| \(\mathrm{ECR}_{\mathrm{fix}}\) | same turns, but verify body = **shared GPT critique** (+ forced wrong close) |
| \(\mathrm{Acc}_{\mathrm{regen}}\) | problem only (no \(y_0\) / verify / regenerate) |
| \(\Delta_{\mathrm{rect}}^{@1}\) | \(\mathrm{ECR}_{\mathrm{fix}}^{@1}-\mathrm{Acc}_{\mathrm{regen}}^{@1}\) |
| \(\Delta_{\mathrm{base}}^{@1}\) | \(\mathrm{ECR}_{\mathrm{fix}}^{@1}\) vs Pre-RL |

## Protocol

1. **Oracle \(a_1\)**: GT grading only (`extract_answer` + `math_equal`). Wrong ⇒ enter pool. No GenRM / no FP/FN from verify.
2. **Fixed pool**: Pre-RL (`Qwen2.5-1.5B-Instruct`) one \(y_0\) per MATH-500 problem; same pool for Pre-RL & PPO.
3. **One-round revise only**, prompts match **PAG RL ChatML** (not S2R bridges):
   `problem → y0 → VERIFY_USER → verify body (… The answer is wrong.) → REGENERATE_USER → y1`.
4. **Fixed critiques**: **GPT API** (`generate_fixed_critiques_gpt.py`); injected as the verify assistant turn and forced to end with `The answer is wrong.`
5. **@1 / @8**: `T=0,n=1` vs `T=0.7,n=8` (pass@8).

## Layout

```
tools/controlled_rectify/
  build_prerl_pool.py
  generate_fixed_critiques_gpt.py   # GPT API → fixed_critique
  controlled_rectify_eval.py
  aggregate_table.py
  run_pipeline.sh
  data/
  results/
```

## Setup

```bash
export OPENAI_API_KEY=...
# optional:
export OPENAI_BASE_URL=...          # Azure / proxy
export GPT_CRITIQUE_MODEL=gpt-5o    # or your GPT-5.x id
export CUDA_VISIBLE_DEVICES=0,1
export TP=2
```

PPO FSDP shards must be merged once:

```bash
bash tools/controlled_rectify/run_pipeline.sh merge
# → checkpoints/PAG/qwen1p5b_pag/global_step_400/actor_hf
```

## Run

```bash
cd /path/to/Policy-As-GenVerifier

# step by step
bash tools/controlled_rectify/run_pipeline.sh pool
bash tools/controlled_rectify/run_pipeline.sh critique   # needs OPENAI_API_KEY
bash tools/controlled_rectify/run_pipeline.sh eval_at1
bash tools/controlled_rectify/run_pipeline.sh eval_at8
bash tools/controlled_rectify/run_pipeline.sh aggregate

# or everything (needs free GPUs + API key)
bash tools/controlled_rectify/run_pipeline.sh all
```

Outputs:

- `results/pag_prerl_at{1,8}.metrics.json`
- `results/pag_ppo400_at{1,8}.metrics.json`
- `results/table_pag_controlled_rectify.tex`

## Note on GPUs / vLLM

Pool + eval need free GPUs. Critique (GPT) is **API-only** and can run while GPUs are busy.

Defaults match PAG train/eval to avoid V1 sampler OOM:

- `VLLM_USE_V1=0` (forced in `run_pipeline.sh`)
- `GPU_MEM_UTIL=0.70`, `MAX_NUM_SEQS=256` (override via env if needed)
