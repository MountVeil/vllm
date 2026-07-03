HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm bench serve \
	--backend openai-chat \
	--base-url http://127.0.0.1:8000 \
	--endpoint /v1/chat/completions \
	--model qwen05b \
	--tokenizer /home/xx/models/hf/Qwen2.5-0.5B-Instruct \
	--dataset-name random \
	--random-input-len 256 \
	--random-output-len 128 \
	--num-prompts 20 \
	--request-ra inf \
	--max-concurrency 2 \
	--result-dir /tmp/results \
	--result-filename Qwen2.5-0,5B-Instruct.json \
	--save-result \
	--percentile-metrics ttft,tpot,itl,e2el \
