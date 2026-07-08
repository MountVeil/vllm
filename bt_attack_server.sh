#!/usr/bin/env bash
set -u

rm -f /home/xx/tee-llm-thor/logs/bt_attack.jsonl
rm -f /home/xx/tee-llm-thor/logs/vllm_bt_defense.jsonl

export TZKV_LOG=/home/xx/tee-llm-thor/logs/bt_attack.jsonl

# ============================================================
# Baseline attack mode:
# TEE BT defense OFF, BT attack ON.
# ============================================================

export TZKV_OPTEE_BT_ENABLE=0
export TZKV_OPTEE_BT_STRICT=0

unset TZKV_OPTEE_BT_REPO
unset TZKV_OPTEE_BT_TOOL
unset TZKV_OPTEE_BT_LIB
unset TZKV_OPTEE_BT_LOG
unset TZKV_OPTEE_BT_RESET_ON_INIT
unset TZKV_OPTEE_BT_COMMIT_MODE
unset TZKV_OPTEE_BT_TAMPER_AFTER_COMMIT

export TZKV_KV_ATTACK_ENABLE=0
export TZKV_DESC_ATTACK_ENABLE=0
export TZKV_BT_ATTACK_ENABLE=1

export TZKV_BT_ATTACK_MAX_BLOCKS=1024
export TZKV_BT_ATTACK_MIN_PREFILL=1
export TZKV_BT_ATTACK_SOURCE_ORDER=0
export TZKV_BT_ATTACK_VICTIM_ORDER=1
export TZKV_BT_ATTACK_ONCE=0

MODEL=/home/xx/models/hf/Qwen2.5-0.5B-Instruct

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen05b \
  --dtype bfloat16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.25 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000
