import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

thrput_dict = {
    # "cloud": (50, 450),
    "orin_50w": (9.5, 80),
    "orin_30w": (7, 64),
    "orin_15w": (4, 60)
}
result_file_dir = "/home/jingcan/workspace/Spec-Bench/data/spec_bench/speed/" 

import matplotlib.pyplot as plt
default_plt_size = (5.5, 3.5)
def init_plot():
    cmap = plt.get_cmap('Set2')
    colors = [cmap(i) for i in range(8)]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    

def create_parent_directory(dir_name):
    parent_directory = dir_name
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print(f"created: {parent_directory}")
    else:
        print(f"existent: {parent_directory}")
subtask_list = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]
def cal_single_dev_speedup_ratio(jsonl_file, jsonl_file_base, device_name, tokenizer):
    # use theory to calculate
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]
    plt.clf()
    plt.cla()
    init_plot()
    draft_throughput_gen_1 = thrput_dict[device_name][1]   # 吞吐量，单位为tokens每秒
    target_thrput_gen_1 = thrput_dict[device_name][0]   # 吞吐量，单位为tokens每秒
    for task in subtask_list:
        
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                if task=="overall":
                    data.append(json_obj)
                elif task == "mt_bench":
                    if json_obj["category"] in mt_bench_list:
                        data.append(json_obj)
                else:
                    if json_obj["category"] == task:
                        data.append(json_obj)

        speeds = []
        accept_lengths_list = []
        for datapoint in data:
            average_latency = 1 / draft_throughput_gen_1
            tokens = sum(datapoint["choices"][0]['new_tokens'])
            ass_list = datapoint["choices"][0]["assited_lengths"]
            acc_list = datapoint["choices"][0]["accept_lengths"]
            actual_times = [tokens * average_latency for tokens in ass_list]
            tot_actual_time = sum(actual_times) 
            est_target_time = 1 / target_thrput_gen_1 * len(ass_list)
            assert(tokens == sum(acc_list) + len(acc_list))
            speed = tokens / (tot_actual_time + est_target_time)
            speeds.append(speed)
            
        data = []
        with open(jsonl_file_base, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                if task=="overall":
                    data.append(json_obj)
                elif task == "mt_bench":
                    if json_obj["category"] in mt_bench_list:
                        data.append(json_obj)
                else:
                    if json_obj["category"] == task:
                        data.append(json_obj)

        total_time = 0
        total_token = 0
        speeds0 = []
        for datapoint in data:
            answer = datapoint["choices"][0]['turns']
            tokens = 0
            for i in answer:
                tokens += (len(tokenizer(i).input_ids) - 1)
            times = 1 / target_thrput_gen_1 * tokens
            speeds0.append(tokens / times)
            
        tokens_per_second = np.array(speeds).mean()
        tokens_per_second_baseline = np.array(speeds0).mean()
        speedup_ratio = np.array(speeds).mean() / np.array(speeds0).mean()
        print("="*30, "Task: ", task, "="*30)
        # print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
        

            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        default='../data/mini_bench/model_answer/vicuna-7b-v1.3-eagle-float32-temperature-0.0.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default='../data/mini_bench/model_answer/vicuna-7b-v1.3-vanilla-float32-temp-0.0.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default='/data/heming/pretrained_models/vicuna-7b-v1.3/',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--mean-report",
        action="store_true",
        default=False,
        help="report mean speedup over different runs")
    parser.add_argument(
        "--device-name",
        default='orin_50w',
        type=str,
        help="device name")

    args = parser.parse_args()
    
    import os
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
    cal_single_dev_speedup_ratio(args.file_path, args.base_path, args.device_name, tokenizer)
   