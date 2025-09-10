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
GPU_DEVICES=0,1,2

bench_NAME="pipespec_bench_0.05_0.03_0.15"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]
shared_suffix="_w3d6_tree"
set -x
num_assistant_tokens=6
mode="pipespec_mp_sm_async${shared_suffix}"
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} IS_SERVER=1 SIMULATE_DEV_DRAFT_TIME=0.05 SIMULATE_SERVER_DRAFT_TIME=0.03 SIMULATE_SERVER_VERIFY_TIME=0.15 TREE_WIDTH=3 TREE_DEPTH=6 NUM_ASSISTED_TOKENS=6 USE_TREE=1 GENERATE_TOKEN_NUM=512 python -m evaluation.inference_pipespec --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 512 --async-mode True --use-tree True > ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens}.log 2>&1

# num_assistant_tokens=6
# mode="pipespec_mp_sm_async${shared_suffix}"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} IS_SERVER=1 SIMULATE_DEV_DRAFT_TIME=0.05 SIMULATE_SERVER_DRAFT_TIME=0.03 SIMULATE_SERVER_VERIFY_TIME=0.15 TREE_WIDTH=4 TREE_DEPTH=6 NUM_ASSISTED_TOKENS=6 USE_TREE=0 GENERATE_TOKEN_NUM=512 python -m evaluation.inference_pipespec --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --num_assistant_tokens ${num_assistant_tokens} --max-new-tokens 512 --async-mode True > ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens}.log 2>&1

num_assistant_tokens=5
# mode="edge_spec${shared_suffix}"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype  --max-new-tokens 512 --use-edgespec > ${MODEL_NAME}-${mode}-${DRAFT_MODEL_NAME}-${torch_dtype}-temp-${TEMP}-asstkn-${num_assistant_tokens}.log 2>&1

