API_KEY=${OPENAI_API_KEY}
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME="Meta-Llama-3-8B-Instruct"
rm -rf ${MODEL_NAME}_vllm_server.log

python -m vllm.entrypoints.openai.api_server \
    --model /netcache/huggingface/${MODEL_NAME} \
    --dtype auto \
    --api-key ${API_KEY} \
    --port 28000 >> ${MODEL_NAME}_vllm_server.log 2>&1 &