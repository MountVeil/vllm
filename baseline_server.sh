MODEL=/home/xx/models/hf/Qwen2.5-0.5B-Instruct
TZKV_JSONL=/home/xx/tee-llm-thor/logs/perf_baseline_vllmbench.log.jsonl
SERVER_STDOUT=/home/xx/tee-llm-thor/logs/perf_baseline_vllmbench.server.out
rm -rf "$TZKV_JSONL" "$SERVER_STDOUT"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
export TZKV_LOG=/home/xx/tee-llm-thor/logs/clean_baseline.jsonl 
export TZKV_BT_ATTACK_ENABLE=0
export TZKV_DESC_ATTACK_ENABLE=0
export TZKV_KV_ATTACK_ENABLE=0
#TZKV_KV_PAYLOAD_MIN_ENABLE=0 \
#TZKV_KV_OPTEE_ENABLE=0 \
#TZKV_BLOCK_TABLE_ATTACK_ENABLE=0 \
#TZKV_BT_OPTEE_ENABLE=0 \
#TZKV_DESCRIPTOR_ATTACK_ENABLE=0 \
#TZKV_DESC_OPTEE_ENABLE=0 \
#TZKV_ROOT=/home/xx/tee-llm-thor \
#TZKV_LOGS="$TZKV_JSONL" \
python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" --served-model-name qwen05b --dtype bfloat16 \
	--max-model-len 512 --gpu-memory-utilization 0.25 \
	--max-num-seqs 2 --max-num-batched-tokens 1024 \
	--enforce-eager --host 127.0.0.1 --port 8000 \

