"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, AutoConfig
from model.sps.decoding import assisted_decoding, new_assisted_decoding
import sys 
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline_spec.src.utils.config import CLIENT_DEVICE, SERVER_URL 
from pipeline_spec.src.model.spec_wrapper import AsyncSpecDecodingWrapper, SpecLlamaForCausalLM
from pipeline_spec.src.model.spec_wrapper import SpecDecodingWrapper
from pipeline_spec.src.model.vice_wrapper import ViceModelWrapper
from model.pipespec.decoding import async_target_generate, target_generate, async_spec_decoding, spec_decoding
from loguru import logger

def pipespec_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None, drafter_tokenizer=None, num_assistant_tokens=4, assistant_confidence_threshold=0.0):
    input_ids = inputs.input_ids
    input_len = len(input_ids[0])
    generation_config = {}
    generation_config["max_new_tokens"] = max_new_tokens
    drafter.max_new_tokens = max_new_tokens
    drafter.generation_config.num_assistant_tokens = num_assistant_tokens
    drafter.generation_config.assistant_confidence_threshold = assistant_confidence_threshold
    # generation_config.pad_token_id = tokenizer.pad_token_id
    drafter.generation_config.pad_token_id = drafter_tokenizer.pad_token_id
    output_ids, idx, accept_length_list, assited_length_list = model.generate(
        **inputs, **generation_config, assistant_model=drafter, do_sample=do_sample)
    new_token = len(output_ids[0]) - input_len
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
    parser.add_argument(
        "--async-mode",
        type=bool,
        default=False,
        help="Whether to use async mode.",
    )
    args = parser.parse_args()

    # GenerationMixin._assisted_decoding = new_assisted_decoding

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    AsyncSpecDecodingWrapper.async_spec_decoding = async_spec_decoding
    SpecDecodingWrapper.spec_decoding = spec_decoding
    ViceModelWrapper.async_target_generate = async_target_generate
    ViceModelWrapper.target_generate = target_generate
    # draft_model_name = "double7/vicuna-68m"
    # target_model_name = "lmsys/vicuna-7b-v1.3"
    draft_tokenizer = AutoTokenizer.from_pretrained(args.drafter_path)
    draft_model = SpecLlamaForCausalLM.from_pretrained(args.drafter_path, torch_dtype="float16").to(CLIENT_DEVICE)

    target_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    target_model_config = AutoConfig.from_pretrained(args.model_path)
    vice_target_model = ViceModelWrapper(use_async=args.async_mode, real_model_config=target_model_config, real_model_is_stateful=False, url=SERVER_URL)
    
    
    if args.async_mode:
        inference_model = AsyncSpecDecodingWrapper(draft_model, vice_target_model)
        
    else:
        inference_model = SpecDecodingWrapper(draft_model, vice_target_model)
        
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
    draft_model.eval()
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    # logger.disable("pipeline_spec")
    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False
        args.temperature = None
    # input_str = """The 23-year-old has had surgery for what the Pro12 club describe as "an ongoing shoulder complaint".
    # Edinburgh estimate that the former Scotland Under-20 and Scotland Sevens player will be sidelined for up to six months.
    # Kennedy, who had loan spells with Glasgow Warriors and London Irish, is under contract until summer 2016.
    # His last appearance for Edinburgh came as a replacement during the 38-20 European Challenge Cup win over Bordeaux-Begles on 23 January.
    # """

    # messages = [
    #     {"role": "user", "content": f"{input_str}"},
    # ]
    # temp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # formatted_input = temp_tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,  # 如果只需要格式化文本，不进行tokenize
    #     add_generation_prompt=True  # 是否添加生成提示
    # )
    # input_str = formatted_input
    # inputs = draft_tokenizer(input_str, return_tensors="pt").to(draft_model.device)
    # print(inputs)
    # output_ids, idx, accept_length_list, assited_length_list = inference_model.generate(
    #     **inputs, do_sample=do_sample, min_new_tokens=12, max_new_tokens=12, num_return_sequences=1, assistant_model=draft_model)
    # else:
    #     do_sample = False
    with torch.inference_mode():
        run_eval(
        model=inference_model,
        tokenizer=tokenizer,
        forward_func=pipespec_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=draft_model,
        temperature=args.temperature,
        do_sample=do_sample,
        drafter_tokenizer=drafter_tokenizer,
        num_assistant_tokens=args.num_assistant_tokens
    )

    reorg_answer_file(answer_file)