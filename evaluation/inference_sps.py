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


def sps_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None, drafter_tokenizer=None, num_assistant_tokens=20):
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens
    drafter.max_new_tokens = max_new_tokens
    drafter.generation_config.num_assistant_tokens = num_assistant_tokens
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    drafter.generation_config.pad_token_id = drafter_tokenizer.pad_token_id
    output_ids, idx, accept_length_list, assited_length_list = model.generate(
        **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, temperature=temperature, tokenizer=tokenizer)
    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list, assited_length_list


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

    # if args.temperature > 0:
    do_sample = True
    # else:
    #     do_sample = False
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