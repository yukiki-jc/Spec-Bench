import os
import re
import math
import matplotlib.pyplot as plt

bench_name = 'pipespec_bench_0.05_0.03_0.15'
log_dir = f'/home/jingcan/workspace/dev_spec/Spec-Bench/data/{bench_name}/speed/'
# 指定要画的 log 文件名（只使用这些，不做全目录扫描）
# for edge end 
# suffix = "_edge_end"
suffix = "_edge_end"
log_list = [
    # f'speed_to_edge_spec-vicuna-7b-v1.3-pipespec_mp_sm_async_w3d6_tree_edge_end-vicuna-68m-float16-temp-0-asstkn-6' 
    # f'speed_to_edge_spec-vicuna-7b-v1.3-pipespec_mp_sm_async_w3d6_tree_edge_end-vicuna-68m-float16-temp-0-asstkn-6'
    f'speed_to_edge_spec-vicuna-7b-v1.3-pipespec_mp_sm_async_w3d6_tree{suffix}-vicuna-68m-float16-temp-0-asstkn-6.log',
    f'speed_to_edge_spec-vicuna-7b-v1.3-pipespec_mp_sm_async{suffix}-vicuna-68m-float16-temp-0-asstkn-6.log',
]
# 与 log_list 一一对应的 legend 标签
legend_label_list = [
    'pipe_tree_w3d6',
    'pipe (no-tree)',
]
# for limcpu1
# # 与 log_list 一一对应的 legend 标签
legend_label_list = [
    'pipe_tree_w3d6',
    'pipe (no-tree)',
]

# 连接 log_dir 和 log_list 中的文件名
log_files = [os.path.join(log_dir, name) for name in log_list]

# 正则
task_pattern = re.compile(r'Task:\s+(\w+)')
ratio_pattern = re.compile(r'Speedup ratio:\s+([0-9.]+)')

# 读取并解析
task_names = []
file_ratios = []  # 按 log_list 顺序的 ratio 列表
for path in log_files:
    if not os.path.exists(path):
        print(f"WARN: log not found: {path}")
        file_ratios.append([])
        continue
    with open(path, 'r') as f:
        txt = f.read()
    tasks = task_pattern.findall(txt)
    ratios = [float(x) for x in ratio_pattern.findall(txt)]
    if not task_names:
        task_names = tasks
    file_ratios.append(ratios)

if not task_names:
    raise SystemExit('No tasks found in logs')

# 规范化每个文件的 ratios 长度，缺失用nan填充
max_len = len(task_names)
for i in range(len(file_ratios)):
    r = file_ratios[i]
    if len(r) < max_len:
        r = r + [math.nan] * (max_len - len(r))
    file_ratios[i] = r

# 画图（更紧凑的布局，字体更大，图片更小）
fig, ax = plt.subplots(figsize=(6, 3))
num_files = len(file_ratios)
indices = list(range(max_len))
# 稍微放宽每组的宽度，让整体更紧凑
bar_width = 0.75 / max(1, num_files)

for i, ratios in enumerate(file_ratios):
    # x 位置
    xs = [x + (i - (num_files - 1) / 2) * bar_width for x in indices]
    # 将 nan 保持为 0（matplotlib 不画 nan），但我们想保留真实高度用于 ylim 计算
    plot_values = [v if (not math.isnan(v)) else 0 for v in ratios]
    ax.bar(xs, plot_values, width=bar_width, label=(legend_label_list[i] if i < len(legend_label_list) else f'log{i}'))

ax.set_xticks(indices)
ax.set_xticklabels(task_names, rotation=30, ha='right', fontsize=12)
ax.set_ylabel('Speedup Ratio', fontsize=13)
# ax.set_title('Speedup Ratio', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
# 减小 x 轴与图像的间距，整体更紧凑
plt.tight_layout(pad=0.3)
ax.tick_params(axis='x', which='major', pad=6)

# y 轴下界设为 0.9，上界取实际数据的 max 与 1.0 的较大值并留点空白
all_vals = [v for arr in file_ratios for v in arr if (not math.isnan(v))]
if all_vals:
    ymax = max(1.0, max(all_vals))
else:
    ymax = 1.0
ax.set_ylim(0.9, 1.15)

# 保存时裁剪多余边距并降低输出尺寸
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, zorder=5)  # 在 y=1 处画一条分界线
ax.text(0.99, 1.0, '1.0', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=10, color='gray')  # 可选地在右上角标注 1.0，靠近坐标轴显示
out = f'speedup_ratio_bar{suffix}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
# plt.show()