import os
import re
import matplotlib.pyplot as plt
import numpy as np 

def extract_speedup_ratio(filepath):
    """从指定日志文件中提取 Speedup ratio 的值。"""
    speedup_ratios = []
    with open(filepath, 'r') as file:
        content = file.readlines() 
        for line in content:
            # 使用正则表达式查找 Speedup ratio
            match = re.search(r'Speedup ratio:\s+([\d.]+)', line)
            if match:
                ratio = float(match.group(1))
                speedup_ratios.append(ratio)  # 保存文件名和对应的 Speedup ratio
    return speedup_ratios 
subtask_list = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]
def get_max_speedup_ratio(file_dir):
    """获取指定文件夹中所有 log 文件的最大 Speedup ratio。"""
    
    log_files = [f'{file_dir}/speed_vicuna-7b-v1.3-sps-vicuna-68m-float16-temp-0.1-asstkn-{k}.log' for k in [1, 2, 3, 4, 5, 10, 15, 20, 25]]
    
    task_best_token_ratio = {}
    task_all_token_ratio = {}
    for i, task in enumerate(subtask_list):
        max_ratio = float('-inf')
        best_token = 0
        task_all_token_ratio[task] = {}
        for filename in log_files:
            token_num = filename.split('-')[-1].split('.')[0]
            ratios = extract_speedup_ratio(filename)
            ratio = ratios[i]
            if ratios is not None:
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_token = token_num
            task_all_token_ratio[task][token_num] = ratio
        task_best_token_ratio[task] = (max_ratio, best_token)
            
    return task_best_token_ratio, task_all_token_ratio


speedup_ratios = []
thrput_dict = {
    # "cloud": (50, 450),
    "orin_50w": (9.5, 80),
    "orin_30w": (7, 73),
    "orin_15w": (4, 62)
}
devices = list(thrput_dict.keys())
device_best_dict = {}
device_all_task_dict = {}
for device_name in devices:
    folder = f"/home/jingcan/workspace/Spec-Bench/data/{device_name}_spec_bench/speed"
    task_best_dict, task_all_dict = get_max_speedup_ratio(folder)
    device_best_dict[device_name] = task_best_dict
    device_all_task_dict[device_name] = task_all_dict
    
def draw_best_token_bars(device_best_dict): 
    scores = np.zeros((len(subtask_list), len(devices)))
    for i, task in enumerate(subtask_list):
        for j, device in enumerate(device_best_dict.keys()):
            scores[i, j] = device_best_dict[device][task][1]  # 默认得分为 0

    # 绘制柱状图
    bar_width = 0.2  # 每个柱子的宽度
    x = np.arange(len(subtask_list))  # 任务的 x 位置

    plt.figure(figsize=(10, 6))

    for i, device in enumerate(devices):
        plt.bar(x + i * bar_width, scores[:, i], width=bar_width, label=device)

    # 设置图表的细节
    plt.xlabel('Tasks')
    plt.ylabel('Best Proposed Length (PL)')
    # plt.title('Best Token Num')
    plt.xticks(x + bar_width / 2, subtask_list)  # 设置 x 轴刻度
    plt.legend()
    plt.tight_layout()  # 使布局更紧凑
    plt.savefig(f"task_ratio_devices.png")
import sys 
sys.path.append("/home/jingcan/workspace/dev_spec")
from set_plot import *

def draw_task_avg_speedup(token_num=4): 
    scores = np.zeros((len(subtask_list), len(devices)))
    for i, task in enumerate(subtask_list):
        for j, device in enumerate(device_best_dict.keys()):
            scores[i, j] = device_all_task_dict[device][task][str(token_num)]  # 默认得分为 0
    plt.figure(figsize=default_plt_size)  # 设置图形大小
    init_plot()
    # 绘制柱状图
    bar_width = 0.2  # 每个柱子的宽度
    x = np.arange(len(subtask_list))  # 任务的 x 位置
    for i, device in enumerate(devices):
        plt.bar(x + i * bar_width, scores[:, i], width=bar_width, label=device)

    # 设置图表的细节
    plt.xlabel('Tasks')
    plt.ylabel('Speedup Ratio')
    # plt.title('Best Token Num')
    plt.xticks(x + bar_width / 2, subtask_list)  # 设置 x 轴刻度
    plt.legend()
    plt.tight_layout()  # 使布局更紧凑
    plt.xticks(rotation=20)
    plt.axhline(y=1, color='red', linestyle='--')
    plt.savefig("task_avg_speedup.png")
draw_task_avg_speedup(4)