rm -f /home/xx/tee-llm-thor/logs/bt_attack.jsonl

export TZKV_LOG=/home/xx/tee-llm-thor/logs/bt_attack.jsonl

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
