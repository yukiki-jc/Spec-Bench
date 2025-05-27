"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding, new_assisted_decoding
import sys, os 
sys.path.append("/home/jingcan/workspace/dev_spec/LLM-Pruner")

def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None, drafter_tokenizer=None, num_assistant_tokens=20):
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens
    drafter.max_new_tokens = max_new_tokens
    drafter.generation_config.num_assistant_tokens = num_assistant_tokens
    drafter.generation_config.assistant_confidence_threshold = 0.0
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    drafter.generation_config.pad_token_id = drafter_tokenizer.pad_token_id
    output_ids, idx, accept_length_list, assited_length_list = model.generate(
        **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, temperature=temperature, tokenizer=tokenizer)
    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list, assited_length_list
from transformers import LlamaConfig, LlamaForCausalLM
def initialize_custom_llama_model(layers, heads, intermediate_size, embedding_size, ref_model=None):
    # 创建自定义的配置
    config = LlamaConfig(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate_size,
        hidden_size=embedding_size
    )
    
    # 初始化模型
    model = LlamaForCausalLM(config)
    if ref_model:
        model.model.embed_tokens = ref_model.model.embed_tokens
        # copy each layers' weight from ref_model
        for i in range(layers):
            model.model.layers[i].self_attn.q_proj.weight = ref_model.model.layers[i].self_attn.q_proj.weight
            model.model.layers[i].self_attn.k_proj.weight = ref_model.model.layers[i].self_attn.k_proj.weight
            model.model.layers[i].self_attn.v_proj.weight = ref_model.model.layers[i].self_attn.v_proj.weight
            model.model.layers[i].self_attn.o_proj.weight = ref_model.model.layers[i].self_attn.o_proj.weight
            model.model.layers[i].mlp.up_proj.weight = ref_model.model.layers[i].mlp.up_proj.weight
            model.model.layers[i].mlp.down_proj.weight = ref_model.model.layers[i].mlp.down_proj.weight
            model.model.layers[i].mlp.gate_proj.weight = ref_model.model.layers[i].mlp.gate_proj.weight
            # and bias 
            model.model.layers[i].self_attn.q_proj.bias = ref_model.model.layers[i].self_attn.q_proj.bias
            model.model.layers[i].self_attn.k_proj.bias = ref_model.model.layers[i].self_attn.k_proj.bias
            model.model.layers[i].self_attn.v_proj.bias = ref_model.model.layers[i].self_attn.v_proj.bias
            model.model.layers[i].self_attn.o_proj.bias = ref_model.model.layers[i].self_attn.o_proj.bias
            model.model.layers[i].mlp.up_proj.bias = ref_model.model.layers[i].mlp.up_proj.bias
            model.model.layers[i].mlp.down_proj.bias = ref_model.model.layers[i].mlp.down_proj.bias
            model.model.layers[i].mlp.gate_proj.bias = ref_model.model.layers[i].mlp.gate_proj.bias
        model.lm_head.weight = ref_model.lm_head.weight
        model.lm_head.bias = ref_model.lm_head.bias
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drafter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--num_assistant_tokens",
        type=int,
        default=20, 
        help="Number of assistant tokens. Default 20.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="The path to the pruned model.",
    )
    args = parser.parse_args()

    GenerationMixin._assisted_decoding = new_assisted_decoding

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    if args.ckpt:
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, drafter = pruned_dict['tokenizer'], pruned_dict['model']
        
        # tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        drafter.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        drafter.config.bos_token_id = 1
        drafter.config.eos_token_id = 2
        drafter = initialize_custom_llama_model(
            layers=drafter.config.num_hidden_layers,
            heads=drafter.config.num_attention_heads,
            intermediate_size=drafter.config.intermediate_size,
            embedding_size=drafter.config.hidden_size,
            ref_model=drafter
        )
        drafter : LlamaForCausalLM = drafter.to("cuda:0").to(torch.float32)
    else:
        drafter = AutoModelForCausalLM.from_pretrained(
            args.drafter_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
    )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    drafter_tokenizer = AutoTokenizer.from_pretrained(args.drafter_path, use_fast=False)
    if "MobileLLM" in args.model_path: 
        tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        })
        drafter_tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        })
        tokenizer.pad_token_id = tokenizer.eos_token_id
        drafter_tokenizer.pad_token_id = drafter_tokenizer.eos_token_id
    model.eval()
    drafter.eval()

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False
        args.temperature = None
    with torch.inference_mode():
        run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=sps_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        temperature=args.temperature,
        do_sample=do_sample,
        drafter_tokenizer=drafter_tokenizer,
        num_assistant_tokens=args.num_assistant_tokens
    )

    reorg_answer_file(answer_file)