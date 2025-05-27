Vicuna_PATH="lmsys/vicuna-7b-v1.3"
# Vicuna_PATH="meta-llama/Llama-2-7b-chat-hf"
# Eagle_PATH=yuhuili/EAGLE-Vicuna-7B-v1.3
# Medusa_PATH=/your_own_path/medusa-vicuna-7b-v1.3
# Hydra_PATH=/your_own_path/hydra-vicuna-7b-v1.3
Drafter_PATH=double7/vicuna-68m
# Drafter_PATH=Jiayi-Pan/Tiny-Vicuna-1B
# Drafter_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Space_PATH=/your_own_path/vicuna-v1.3-7b-space
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx
# MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_NAME="vicuna-7b-v1.3"
DRAFT_MODEL_NAME="vicuna-68m"
# DRAFT_MODEL_NAME="Tiny-Vicuna-1B"
# DRAFT_MODEL_NAME="TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME="Llama-2-7b-chat-hf"
TEMP=0
GPU_DEVICES=0

bench_NAME="pipespec_bench_dev"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --max-new-tokens 1024
set -x
for num_assistant_tokens in 4
do 
    # CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 1024

    # CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps_pruned-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 1024 --ckpt /home/jingcan/workspace/dev_spec/LLM-Pruner/prune_log/vicuna_68m_prune_0.9/pytorch_model.bin

    # CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps_pruned_spec-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 1024 --ckpt /home/jingcan/workspace/dev_spec/LLM-Pruner/prune_log/vicuna_68m_prune_spec_0.9/pytorch_model.bin

    # CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pipespec --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-pipespec-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 1024 > ${MODEL_NAME}-pipespec-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens}.log 2>&1

    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pipespec --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-pipespec_async-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 1024 --async-mode True > ${MODEL_NAME}-pipespec_async-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens}.log 2>&1
done
# Drafter_PATH="Qwen/Qwen2.5-1.5B-Instruct"
# DRAFT_MODEL_NAME=Qwen2.5-1.5B-Instruct
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_hydra --model-path $Hydra_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-hydra-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_space --model-path $Space_PATH --model-id ${MODEL_NAME}-space-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype

