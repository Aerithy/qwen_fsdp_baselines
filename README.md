# Qwen FSDP baselines

This directory provides two FSDP pretraining baselines:

- Qwen-70B pretraining
- Qwen2.5-14B-Instruct pretraining (baseline run as causal LM)

Both use the same reference pretraining setup:

- Model: decoder-only Transformer, RMSNorm, SwiGLU, RoPE, GQA
- Objective: causal next-token prediction
- Optimizer: AdamW (beta1=0.9, beta2=0.95, wd=0.1), grad clip 1.0
- LR schedule: cosine decay with 2% warmup (by tokens), peak LR 2e-4
- Precision: BF16, FlashAttention when available
- Data pipeline: boilerplate removal, language-ID and ratio control,
  exact+near dedup (MinHash/SimHash), quality filtering, PII filtering,
  documented sampling weights

## GPU inventory (ixsmi)

This machine exposes 16 GPUs (BI-V150). The run scripts auto-detect
GPU count with torch.

## Layout

- train_fsdp_pretrain.py: main training entry point
- configs/: per-task YAML config
- scripts/: per-task torchrun launchers

## Usage

1) Baseline uses FineWeb (HuggingFaceFW/fineweb) by default. Update configs if needed.
2) Run a task:

./scripts/run_qwen70b_fsdp.sh
./scripts/run_qwen2_5_14b_instruct_fsdp.sh

Dependencies (install in your env as needed):

pip install torch transformers datasets pyyaml

## Notes

- These are baseline scripts and do not include full production data
  preprocessing. The data pipeline described above is assumed to be
  performed offline.
- If you need sequence packing, multiple datasets, or sampling weights,
  extend train_fsdp_pretrain.py accordingly.
