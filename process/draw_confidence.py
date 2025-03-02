import re

def extract_confidence_from_log(file_path):
    """从日志文件中提取所有包含 'confidence' 的行中的浮点数。"""
    reject_confidence_values = []
    accept_confidence_values = []
    # 打开并读取日志文件
    start = 0
    with open(file_path, 'r') as file:
        for line in file:
            # 查找包含 'confidence' 的行
            if "Warmup done" in line:
                start = 1
            if start == 0: 
                continue
            if 'confidence' in line:
                # 使用正则表达式提取末尾的浮点数
                match = re.search(r'.*confidence.*?([\d.]+)', line)
                if match:
                    confidence = float(match.group(1))
                if "accept" in line:
                    # 将找到的浮点数值转换为 float 并添加到列表中
                    accept_confidence_values.append(confidence)
                else: 
                    reject_confidence_values.append(confidence)            

    return accept_confidence_values, reject_confidence_values


def extract_confidence_from_log_task(file_path):
    """从日志文件中提取所有包含 'confidence' 的行中的浮点数。"""
    total_reject = []
    total_accept = []
    reject_confidence_values = []
    accept_confidence_values = []
    # 打开并读取日志文件
    start = 0
    with open(file_path, 'r') as file:
        for line in file:
            # 查找包含 'confidence' 的行
            if "Warmup done" in line:
                start = 1
            if start == 0: 
                continue
            if "step 1" in line:
                if len(reject_confidence_values):
                    total_reject.append(reject_confidence_values)
                    total_accept.append(accept_confidence_values)
                reject_confidence_values = []
                accept_confidence_values = [] 
            if 'confidence' in line:
                # 使用正则表达式提取末尾的浮点数
                match = re.search(r'.*confidence.*?([\d.]+)', line)
                if match:
                    confidence = float(match.group(1))
                if "accept" in line:
                    # 将找到的浮点数值转换为 float 并添加到列表中
                    accept_confidence_values.append(confidence)
                else: 
                    reject_confidence_values.append(confidence)            

    return total_accept, total_reject

import numpy as np 
import matplotlib.pyplot as plt
def draw_confidence_hist(confidence, tag):
    plt.clf()
    plt.cla()
    bins = np.arange(0, 1.05, 0.05)  # 从0到1.05，步长为0.05
    hist, edges = np.histogram(confidence, bins=bins)

    # 计算每个 bin 的概率
    probabilities = hist / hist.sum()
    bar_width = 0.05
    # 绘制直方图
    plt.bar(edges[:-1] + bar_width / 2, probabilities, width=bar_width, color='skyblue', edgecolor='black', alpha=0.7)

    # 添加标题和标签
    plt.title('Histogram of Confidence Values')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置x轴刻度
    plt.tight_layout()
    plt.savefig(f"confidence_hist_{tag}.png")
# 示例使用
  # 替换为你的日志文件路径
def draw_confidence(log_file_path, model):
    accept_confidence_values, reject_confidence_values = extract_confidence_from_log(log_file_path)

    draw_confidence_hist(accept_confidence_values, f"accept_{model}")
    draw_confidence_hist(reject_confidence_values, f"reject_{model}")

# model = "vicuna1b"
# log_file_path = f'/home/jingcan/workspace/Spec-Bench/motivation_{model}.log'
# draw_confidence(log_file_path, model)
def draw_confidence_task(log_file_path, model):
    total_accept, total_reject = extract_confidence_from_log_task(log_file_path)
    task_idx = [0, 2] 
    for i in range(len(task_idx)):
        last = None if i == len(task_idx) - 1 else task_idx[i + 1]
        cur_task_accept = total_accept[task_idx[i]:last]
        cur_task_reject = total_reject[task_idx[i]:last]
        accept_confidence_values = []
        reject_confidence_values = []
        for j in range(len(cur_task_accept)):
            accept_confidence_values += cur_task_accept[j]
            reject_confidence_values += cur_task_reject[j]
        draw_confidence_hist(accept_confidence_values, f"accept_{model}_task_{i}")
        draw_confidence_hist(reject_confidence_values, f"reject_{model}_task_{i}")
model = "vicuna68m"
log_file_path = f'/home/jingcan/workspace/Spec-Bench/motivation_{model}_task.log'
draw_confidence_task(log_file_path, model)