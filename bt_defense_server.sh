#!/usr/bin/env bash
set -u

rm -f /home/xx/tee-llm-thor/logs/bt_attack.jsonl
rm -f /home/xx/tee-llm-thor/logs/vllm_bt_defense.jsonl

export TZKV_LOG=/home/xx/tee-llm-thor/logs/bt_attack.jsonl

# ============================================================
# Defense-only mode:
# TEE BT defense ON, all attacks OFF.
# Prefix caching disabled because PREFIX_REUSE is not implemented yet.
# ============================================================

unset TZKV_OPTEE_BT_COMMIT_MODE
unset TZKV_OPTEE_BT_TAMPER_AFTER_COMMIT

export TZKV_OPTEE_BT_ENABLE=1
export TZKV_OPTEE_BT_STRICT=1
export TZKV_OPTEE_BT_REPO=/home/xx/tee-llm-thor
export TZKV_OPTEE_BT_TOOL=/home/xx/tee-llm-thor/ca/tzkv_tool
export TZKV_OPTEE_BT_LIB=/home/xx/tee-llm-thor/ca/libtzkv_client.so
export TZKV_CLIENT_BACKEND=auto
export TZKV_OPTEE_BT_LOG=/home/xx/tee-llm-thor/logs/vllm_bt_defense.jsonl
export TZKV_OPTEE_BT_RESET_ON_INIT=0
export TZKV_OPTEE_BT_BLOCK_SIZE=16
export TZKV_OPTEE_BT_MAX_ENTRIES=128

export TZKV_KV_ATTACK_ENABLE=0
export TZKV_DESC_ATTACK_ENABLE=0
export TZKV_BT_ATTACK_ENABLE=0

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
  --no-enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
