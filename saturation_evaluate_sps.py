import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time 
def generate_one_batch(bs, generate_length, drafter):
    prompts = ["Once" for i in range(bs)]
    batch_inputs = tokenizer.batch_encode_plus(prompts, return_tensors='pt').to(device)
    torch.cuda.synchronize()
    s = time.perf_counter()
    output = model.generate(
        batch_inputs.input_ids,
        min_length=generate_length,
        max_length=generate_length,  # 输出的最大长度
        num_return_sequences=1,  # 返回的序列数量
        assistant_model=drafter,
        tokenizer=tokenizer,
        # assistant_tokenizer=tokenizer
        # no_repeat_ngram_size=2,  # 防止重复的 n-gram
        # early_stopping=True
    )
    torch.cuda.synchronize()
    e = time.perf_counter()
    generated_tokens = generate_length * bs 
    print(f"{bs=} {generate_length=} throughput={generated_tokens / (e - s):.2f} tokens/s")
# 加载预训练模型和分词器
model_name = "lmsys/vicuna-7b-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16")
draft_model = "double7/vicuna-68m"
drafter = AutoModelForCausalLM.from_pretrained(draft_model, torch_dtype="float16")
# 你可以选择使用 GPU 加速
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
drafter.to(device)
bss = [1]
# warm up 
generate_one_batch(1, 100, drafter)
# bss = [2]
# generate_lengths = [32, 64, 100, 128, 256, 512, 1024, 2048]
generate_lengths = [32, 64, 128, 256, 512, 1024, 2048]
# 输入提示

with torch.inference_mode():
    for bs in bss:
        for generate_length in generate_lengths:
            generate_one_batch(bs, generate_length, drafter)
# 解码输出
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)