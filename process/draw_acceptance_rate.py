import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

thrput_dict = {
    # "cloud": (50, 450),
    "orin_50w": (9.5, 80),
    # "orin_30w": (7, 73),
    # "orin_15w": (4, 62)
}
result_file_dir = "/home/jingcan/workspace/Spec-Bench/data/spec_bench/speed/" 

def speed(jsonl_file, task=None):
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

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

    accept_lengths_list = []
    assited_lengths_list = []
    total_wall_time = 0
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        total_wall_time += times
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        assited_lengths_list.extend(datapoint["choices"][0]['assited_lengths'])
    
    return accept_lengths_list, assited_lengths_list, total_wall_time
subtask_list = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]
def report_speed(jsonl_file, jsonl_file_base, tokenizer, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]
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

        speeds=[]
        accept_lengths_list = []
        for datapoint in data:
            tokens=sum(datapoint["choices"][0]['new_tokens'])
            times = sum(datapoint["choices"][0]['wall_time'])
            accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
            speeds.append(tokens/times)


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

        total_time=0
        total_token=0
        speeds0=[]
        for datapoint in data:
            answer=datapoint["choices"][0]['turns']
            tokens = 0
            for i in answer:
                tokens += (len(tokenizer(i).input_ids) - 1)
            times = sum(datapoint["choices"][0]['wall_time'])
            speeds0.append(tokens / times)
            total_time+=times
            total_token+=tokens

        

    

def draw_tokens(task_name, accepted_lengths_list, assisted_lengths_list, tokens, save_dir): 
    import matplotlib.pyplot as plt
    plt.clf()
    plt.cla()
    plt.hist(accepted_lengths_list, bins=max(assisted_lengths_list), alpha=0.7, label='accepted')
    plt.hist(assisted_lengths_list, bins=max(assisted_lengths_list), alpha=0.7, label='assisted')
    plt.title(f'{task_name} accepted and assisted lengths')
    plt.legend()
    plt.savefig(f"{save_dir}/{task_name}_accepted_assisted_lengths_{tokens}.png")



def get_single_speedup(jsonl_file):
    subtask_list_dict = {}
    for subtask_name in subtask_list:
        accept_lengths_list, assited_lengths_list, total_wall_time = speed(jsonl_file,  task=subtask_name)
        subtask_list_dict[subtask_name] = {"accept_lengths_list": accept_lengths_list, "assisted_lengths_list": assited_lengths_list, "total_wall_time": total_wall_time}
    return subtask_list_dict
    
def draw_overall_acceptance_rate(data_dict, save_dir):
    

    xs = list(data_dict.keys())
    
    for sub_task_name in subtask_list:
        avg_accepted_tokens = []  # avg accepted tokens 数据
        avg_assisted_tokens = []  # avg assisted tokens 数据
        avg_rate = []
        for x in xs:
            acc_list = data_dict[x][sub_task_name]["accept_lengths_list"]
            ass_list = data_dict[x][sub_task_name]["assisted_lengths_list"]
            avg_accepted_tokens.append(np.mean(acc_list))
            avg_assisted_tokens.append(np.mean(ass_list))
            print(f"debug {x=}")
            rates = [acc / ass if ass else 0 for acc, ass in zip(acc_list, ass_list)]
            avg_rate.append(np.mean(rates))
            
    # 创建图形和轴
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制 avg accepted tokens 折线
        color = 'tab:blue'
        ax1.set_xlabel('Tokens Size')
        ax1.set_ylabel('Average Tokens', color=color)
        ax1.plot(xs, avg_accepted_tokens, marker='o', color=color, label='Avg Accepted Tokens')
        ax1.plot(xs, avg_assisted_tokens, marker='o', color='tab:green', label='Avg Assisted Tokens')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 20)
        ax1.set_yticks(np.arange(0, 21, 1))
        # 添加图例
        ax1.legend(loc='upper left')

        # 创建 twin axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Acceptance Rate', color=color)
        ax2.plot(xs, avg_rate, marker='o', color=color, linestyle='--', label='Avg Acceptance Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks(np.arange(0, 0.42, 0.02))
        ax2.set_ylim(0, 0.4)
        ax1.grid(True)
        plt.legend()
        plt.savefig(f"{save_dir}/acc_rate_{sub_task_name}.png")

import matplotlib.pyplot as plt
import sys 
sys.path.append("/home/jingcan/workspace/dev_spec")
from set_plot import *
    

def create_parent_directory(dir_name):
    parent_directory = dir_name
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print(f"created: {parent_directory}")
    else:
        print(f"existent: {parent_directory}")

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
        
def draw_waste_time(data_dict, device_name, save_dir): 
    plt.clf()
    plt.cla()
    init_plot()
    draft_throughput_gen_1 = thrput_dict[device_name][1]   # 吞吐量，单位为tokens每秒
    target_thrput_gen_1 = thrput_dict[device_name][0]   # 吞吐量，单位为tokens每秒
    xs = list(data_dict.keys()) # expected_tokens
    xs = ['1', '2', '3', '4', '5', '10', '15', '20', '25']
    
    for x in xs:
        wall_times = [100 for _ in subtask_list]
        waste_percents = []
        expect_draft_percents = []
        estimated_target_model_percents = []
        for sub_task_name in subtask_list:
        
            acc_list = data_dict[x][sub_task_name]["accept_lengths_list"]
            ass_list = data_dict[x][sub_task_name]["assisted_lengths_list"]
            tot_wall_time = data_dict[x][sub_task_name]["total_wall_time"]
            
            average_latency = 1 / draft_throughput_gen_1
            expected_times = [tokens * average_latency for tokens in acc_list]
            actual_times = [tokens * average_latency for tokens in ass_list]
            
            tot_expected_time = sum(expected_times)
            tot_actual_time = sum(actual_times) 
            est_target_time = 1 / target_thrput_gen_1 * len(ass_list)
            act_target_time = tot_wall_time - tot_actual_time
            print(f"debug {est_target_time=} {act_target_time=}")
            if device_name != "cloud":
                tot_wall_time = tot_wall_time
                
            waste_draft_time_percentage = (tot_actual_time - tot_expected_time) / tot_wall_time * 100
            waste_percents.append(waste_draft_time_percentage)
            
            expect_draft_time_percentage = (tot_expected_time) / tot_wall_time * 100
            expect_draft_percents.append(expect_draft_time_percentage)
            
            
            # estimated_target_model_percentage = est_target_time / tot_wall_time * 100
            # estimated_target_model_percents.append(estimated_target_model_percentage)
        bar_width = 0.45
        plot_x = np.arange(len(subtask_list))
        fig_size = list(default_plt_size)
        fig, ax = plt.subplots(figsize=tuple(fig_size))
        ax.bar(plot_x, wall_times, width=bar_width, label='verification (%)')

        # 绘制浪费时间百分比柱状图（在实际时间柱状图上叠加）
        ax.bar(plot_x, waste_percents, width=bar_width, label='waste draft (%)')
        ax.bar(plot_x, expect_draft_percents, bottom=waste_percents, width=bar_width, label='valid draft (%)')
        # ax.bar(x, estimated_target_model_percentage, bottom=wall_times, width=bar_width, label='Est Time (%)', color='orange') # est time almost same as actual llm time

        # 设置x轴和y轴
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Device {device_name}')
        ax.set_xticks(plot_x)
        ax.set_xticklabels(subtask_list, rotation=15)

        # 添加图例
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/waste_time_{device_name}_{x}.png")
            
            
if __name__ == "__main__":
    import os
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
    for device_name in thrput_dict.keys():
        
        
        root_dir = f"/home/jingcan/workspace/Spec-Bench/data/{device_name}_spec_bench"
        json_file_dir = f"{root_dir}/model_answer"
        # 定义需要处理的log文件名
        
        result_file_dir = f"{root_dir}/speed" 
        # temporary file for orin 
        if "orin" in device_name:
            root_dir = f"/home/jingcan/workspace/Spec-Bench/data/orin_50w_spec_bench"
            json_file_dir = f"{root_dir}/model_answer"
            result_file_dir = f"/home/jingcan/workspace/Spec-Bench/data/{device_name}_spec_bench/speed"
            
        base_file = f'{json_file_dir}/vicuna-7b-v1.3-vanilla-float16-temp-0.1.jsonl'
        log_files = [f'{json_file_dir}/vicuna-7b-v1.3-sps-vicuna-68m-float16-temp-0.1-asstkn-{k}.jsonl' for k in [1, 2, 3, 4, 5, 10, 15, 20, 25]]      
        create_parent_directory(result_file_dir)    
        
        data_dict = {}
        for log_file in log_files:
            tokens = log_file.split('-')[-1].split('.')[0]
            subtask_list_dict = get_single_speedup(jsonl_file=log_file)
            for key in subtask_list_dict.keys():
                accept_lengths_list = subtask_list_dict[key]["accept_lengths_list"]
                assited_lengths_list = subtask_list_dict[key]["assisted_lengths_list"]
                subtask_name = key
                draw_tokens(subtask_name, accept_lengths_list, assited_lengths_list, tokens, save_dir=result_file_dir)
            data_dict[tokens] = subtask_list_dict
        # draw_overall_acceptance_rate(data_dict)
        
        draw_waste_time(data_dict, device_name, save_dir=result_file_dir) 