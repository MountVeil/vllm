pkill -f "vllm.entrypoints.openai.api_server" || true
pkill -f "EngineCore" || true

/home/xx/tee-llm-thor/ca/tzkv_tool reset || true

MODEL=/home/xx/models/hf/Qwen2.5-0.5B-Instruct
TZKV_JSONL=/home/xx/tee-llm-thor/logs/perf_all_guard_vllmbench.log.jsonl
SERVER_STDOUT=/home/xx/tee-llm-thor/logs/perf_all_guard_vllmbench.server.out
rm -f "$TZKV_JSONL" "$SERVER_STDOUT"

TZKV_TIMING_ENABLE=1 \
TZKV_LOG_MODE=jsonl \
TZKV_META_PREPARE_ENABLE=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TZKV_CLIENT_BACKEND=ctypes \
TZKV_OPTEE_LIB=/home/xx/tee-llm-thor/ca/libtzkv_client.so \
TZKV_KV_PAYLOAD_MIN_ENABLE=0 \
TZKV_BLOCK_TABLE_ATTACK_ENABLE=0 \
TZKV_DESCRIPTOR_ATTACK_ENABLE=0 \
TZKV_KV_OPTEE_ENABLE=1 \
TZKV_KV_OPTEE_STRICT=1 \
TZKV_KV_MODE=committed \
TZKV_KV_OPTEE_RESET=0 \
TZKV_KV_ON_MISMATCH=repair \
TZKV_BT_OPTEE_ENABLE=1 \
TZKV_BT_OPTEE_STRICT=1 \
TZKV_BT_OPTEE_RESET=0 \
TZKV_DESC_OPTEE_ENABLE=1 \
TZKV_DESC_OPTEE_STRICT=1 \
TZKV_DESC_OPTEE_RESET=0 \
TZKV_OPTEE_TOOL=/home/xx/tee-llm-thor/ca/tzkv_tool \
TZKV_ROOT=/home/xx/tee-llm-thor \
TZKV_LOG="$TZKV_JSONL" \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen05b --dtype bfloat16 \
  --max-model-len 512 --gpu-memory-utilization 0.25 \
  --max-num-seqs 2 --max-num-batched-tokens 1024 \
  --enforce-eager --host 127.0.0.1 --port 8000 \
