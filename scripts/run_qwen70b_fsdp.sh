#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/ytli_test/bin/python"
CONFIG="$ROOT_DIR/configs/qwen70b_pretrain.yaml"

get_cfg() {
  "$PYTHON_BIN" - "$CONFIG" "$1" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
key = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
value = cfg.get(key)
if value is None and key == "tokenizer_name":
    value = cfg.get("model_name")
print("" if value is None else value)
PY
}

get_flag() {
  "$PYTHON_BIN" - "$CONFIG" "$1" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
key = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
flag = key.replace("_", "-")
print(f"--{flag}" if cfg.get(key) else "")
PY
}

NPROC="$($PYTHON_BIN - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

TORCHRUN_ARGS=("--nproc_per_node=$NPROC")

"$PYTHON_BIN" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  "$ROOT_DIR/train_fsdp_pretrain.py" \
  --model-name "$(get_cfg model_name)" \
  --tokenizer-name "$(get_cfg tokenizer_name)" \
  --dataset-name-or-path "$(get_cfg dataset_name_or_path)" \
  --dataset-config "$(get_cfg dataset_config)" \
  --text-field "$(get_cfg text_field)" \
  --seq-len "$(get_cfg seq_len)" \
  --per-device-batch-size "$(get_cfg per_device_batch_size)" \
  --grad-accum-steps "$(get_cfg grad_accum_steps)" \
  --lr "$(get_cfg lr)" \
  --warmup-ratio "$(get_cfg warmup_ratio)" \
  --max-tokens "$(get_cfg max_tokens)" \
  --max-steps "$(get_cfg max_steps)" \
  --weight-decay "$(get_cfg weight_decay)" \
  --beta1 "$(get_cfg beta1)" \
  --beta2 "$(get_cfg beta2)" \
  --clip-norm "$(get_cfg clip_norm)" \
  --log-interval "$(get_cfg log_interval)" \
  --save-interval "$(get_cfg save_interval)" \
  --save-dir "$(get_cfg save_dir)" \
  --num-workers "$(get_cfg num_workers)" \
  $(get_flag use_flash_attn) \
  $(get_flag bf16) \
  $(get_flag fp16) \
  $(get_flag activation_checkpointing)
