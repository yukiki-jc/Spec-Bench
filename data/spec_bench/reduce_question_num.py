import json
import random

# 输入和输出文件路径
input_file_path = 'question.jsonl'
output_file_path = 'question_small.jsonl'

# 读取 JSONL 文件
data = {}
with open(input_file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        category = item['category']
        if category not in data:
            data[category] = []
        data[category].append(item)

# 选择每个类别中的 2 个元素
selected_items = []
for category, items in data.items():
    selected = random.sample(items, min(2, len(items)))  # 确保不会超过可用元素数量
    selected_items.extend(selected)

# 写入新的 JSONL 文件
with open(output_file_path, 'w') as f:
    for item in selected_items:
        f.write(json.dumps(item) + '\n')

print(f'Selected items have been written to {output_file_path}')