# KV payload attack
MODEL=/home/xx/models/hf/Qwen2.5-0.5B-Instruct
LOG=/home/xx/tee-llm-thor/logs/kv_payload_min_vllm014.log.jsonl
rm -f "$LOG"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TZKV_KV_PAYLOAD_MIN_ENABLE=1 \
TZKV_BLOCK_TABLE_ATTACK_ENABLE=0 \
TZKV_KV_PAYLOAD_MIN_MAX_TOKENS=128 \
TZKV_KV_PAYLOAD_MIN_MIN_PREFILL=8 \
TZKV_KV_PAYLOAD_MIN_ONCE=1 \
TZKV_BLOCK_TABLE_ATTACK_ENABLE=0 \
TZKV_LOG="$LOG" \
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


# Block table attack
MODEL=/home/xx/models/hf/Qwen2.5-0.5B-Instruct
LOG=/home/xx/tee-llm-thor/logs/block_table_attack_vllm014.log.jsonl
rm -f "$LOG"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TZKV_KV_PAYLOAD_MIN_ENABLE=0 \
TZKV_BLOCK_TABLE_ATTACK_ENABLE=1 \
TZKV_BLOCK_TABLE_ATTACK_MAX_BLOCKS=16 \
TZKV_BLOCK_TABLE_ATTACK_MIN_PREFILL=8 \
TZKV_BLOCK_TABLE_ATTACK_ONCE=0 \
TZKV_LOG="$LOG" \
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
