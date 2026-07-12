#!/usr/bin/env bash
set -euo pipefail
export TZKV_RUNTIME_PROTOCOL="legacy"

export PYTHONPATH="/home/xx/tee-llm-thor/python:/home/xx/vllm:${PYTHONPATH:-}"
export TZKV_RUNTIME_BACKEND=optee
export TZKV_CLIENT_LIBRARY="/home/xx/tee-llm-thor/ca/build/libtzkv_client.so"

MODE="all_defenses"
MODEL="/home/xx/models/hf/Qwen2.5-0.5B-Instruct"
LOG_DIR="/home/xx/tee-llm-thor/logs"

mkdir -p "$LOG_DIR"

export TZKV_LOG="$LOG_DIR/${MODE}_attack.jsonl"
export TZKV_HOOK_LOG="$LOG_DIR/${MODE}_defense.jsonl"
export TZKV_HOOK_LOG_FULL_VALUES=0
export TZKV_HOOK_LOG_ENABLE=0
export TZKV_PROFILE_ENABLE=1
export TZKV_PROFILE_INTERVAL=256
export TZKV_DETAIL_PROFILE_ENABLE=1
export TZKV_DETAIL_PROFILE_INTERVAL=256

rm -f "$TZKV_LOG" "$TZKV_HOOK_LOG"

# Disable discarded legacy paths.
export TZKV_BT_DEFENSE_ENABLE=0
export TZKV_OPTEE_BT_ENABLE=0
export TZKV_OPTEE_BT_STRICT=0
export TZKV_DESC_ENABLE=0
export TZKV_DESC_STRICT=0
export TZKV_DESC_DEFENSE_ENABLE=0
export TZKV_KV_ATTACK_ENABLE=0

unset TZKV_OPTEE_BT_REPO
unset TZKV_OPTEE_BT_TOOL
unset TZKV_OPTEE_BT_LIB
unset TZKV_OPTEE_BT_LOG
unset TZKV_OPTEE_BT_RESET_ON_INIT
unset TZKV_OPTEE_BT_COMMIT_MODE
unset TZKV_OPTEE_BT_TAMPER_AFTER_COMMIT
unset TZKV_CLIENT_BACKEND

# Keep attack parameters defined, but disable synthetic attacks.
export TZKV_BT_ATTACK_MAX_BLOCKS=1024
export TZKV_BT_ATTACK_MIN_PREFILL=1
export TZKV_BT_ATTACK_SOURCE_ORDER=0
export TZKV_BT_ATTACK_VICTIM_ORDER=1
export TZKV_BT_ATTACK_ONCE=0

export TZKV_DESC_ATTACK_MAX_BLOCKS=1024
export TZKV_DESC_ATTACK_SOURCE_ORDER=0
export TZKV_DESC_ATTACK_VICTIM_ORDER=1
export TZKV_DESC_ATTACK_ONCE=0

# Enable the complete current runtime protection.
export TZKV_HOOK_ENABLE=1
export TZKV_HOOK_STRICT=1

# Performance mode: no synthetic attack.
export TZKV_BT_ATTACK_ENABLE=0
export TZKV_DESC_ATTACK_ENABLE=0

echo "============================================================"
echo "Mode:                 $MODE"
echo "Runtime backend:      $TZKV_RUNTIME_BACKEND"
echo "Client library:       $TZKV_CLIENT_LIBRARY"
echo "BT attack:            $TZKV_BT_ATTACK_ENABLE"
echo "Descriptor attack:    $TZKV_DESC_ATTACK_ENABLE"
echo "Runtime defense:      $TZKV_HOOK_ENABLE"
echo "Strict mode:          $TZKV_HOOK_STRICT"
echo "Attack log:           $TZKV_LOG"
echo "Defense log:          $TZKV_HOOK_LOG"
echo "============================================================"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen05b \
  --dtype bfloat16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.25 \
  --enable-prefix-caching \
  --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000
