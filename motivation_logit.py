import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 
from fastchat.conversation import get_conv_template
def generate_one_batch(bs, generate_length, drafter, model_id):
    raw_prompts = ["Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." for i in range(bs)]
    prompts = []
    conv = get_conv_template("qwen-7b-chat")
    for raw_prompt in raw_prompts: 
        # print(f"debug {conv=}")
        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], None)
        conv.stop_str = "</s>"
        prompt = conv.get_prompt()
        prompts.append(prompt)
    batch_inputs = tokenizer.batch_encode_plus(prompts, return_tensors='pt').to(device)
    drafter.generation_config.num_assistant_tokens = 10
    torch.cuda.synchronize()
    s = time.perf_counter()
    output = model.generate(
        batch_inputs.input_ids,
        min_new_tokens=generate_length,
        max_new_tokens=generate_length,  # 输出的最大长度
        num_return_sequences=1,  # 返回的序列数量
        assistant_model=drafter,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.1,
        do_sample=True,
        # assistant_tokenizer=tokenizer
    )
    torch.cuda.synchronize()
    e = time.perf_counter()
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    generated_tokens = generate_length * bs 
    print(f"{bs=} {generate_length=} throughput={generated_tokens / (e - s):.2f} tokens/s")
# 加载预训练模型和分词器
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
draft_model = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

drafter = AutoModelForCausalLM.from_pretrained(
    draft_model,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
drafter_tokenizer = AutoTokenizer.from_pretrained(draft_model, use_fast=False)
if "MobileLLM" in model_name: 
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
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
drafter.to(device)
bss = [1]
model_id = model_name.split('/')[-1]
# warm up 
generate_one_batch(1, 100, drafter, model_id)
# bss = [2]
# generate_lengths = [32, 64, 100, 128, 256, 512, 1024, 2048]
generate_lengths = [1]
# 输入提示

with torch.inference_mode():
    for bs in bss:
        for generate_length in generate_lengths:
            generate_one_batch(bs, generate_length, drafter, model_id)
# 解码输出
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)