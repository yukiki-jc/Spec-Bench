import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 
from fastchat.conversation import get_conv_template
def generate_one_batch(bs, generate_length):
    raw_prompts = ["Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." for i in range(bs)]
    prompts = []
    conv = get_conv_template("zero_shot")
    for raw_prompt in raw_prompts: 
        # print(f"debug {conv=}")
        conv.append_message(conv.roles[0], raw_prompt)
        conv.append_message(conv.roles[1], None)
        conv.stop_str = "</s>"
        prompt = conv.get_prompt()
        prompts.append(prompt)
    batch_inputs = tokenizer.batch_encode_plus(prompts, return_tensors='pt').to(device)
    torch.cuda.synchronize()
    s = time.perf_counter()
    output = model.generate(
        batch_inputs.input_ids,
        min_new_tokens=generate_length,
        max_new_tokens=generate_length,  # 输出的最大长度
        num_return_sequences=1,  # 返回的序列数量
        temperature=0.0,
        do_sample=False,
        # no_repeat_ngram_size=2,  # 防止重复的 n-gram
        # early_stopping=True
    )
    torch.cuda.synchronize()
    e = time.perf_counter()
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    generated_tokens = generate_length * bs 
    print(f"{bs=} {generate_length=} throughput={generated_tokens / (e - s):.2f} tokens/s")
# 加载预训练模型和分词器
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "lmsys/vicuna-7b-v1.3"
model_name = "double7/vicuna-68m"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if "MobileLLM" in model_name: 
    tokenizer.add_special_tokens(
    {
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    })

# 你可以选择使用 GPU 加速
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# bss = [2, 4, 8, 16, 32, 64, 128, 256, 512]
bss = [1]
# warm up 
generate_one_batch(1, 100)
# bss = [2]
generate_lengths = [32, 64, 100, 128, 256, 512, 1024]
# generate_lengths = [320]
# 输入提示

with torch.inference_mode():
    for bs in bss:
        for generate_length in generate_lengths:
            generate_one_batch(bs, generate_length)
# 解码输出
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)
