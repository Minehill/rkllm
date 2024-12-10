# -*- coding: utf-8 -*-

import json

# 读取 JSONL 文件
input_file = '/userdata/repos/datasets/rkllm_code/rknn-llm/rkllm-runtime/examples/rkllm_api_demo/src/dev_rand_split.jsonl'
output_file = 'converted_data.json'

data = []
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        item = json.loads(line)
        question_stem = item['question']['stem']
        choices = item['question']['choices']
        
        # 构建选项描述
        choices_text = "\n".join([f"{choice['label']}: {choice['text']}" for choice in choices])
        
        # 构建 input 字段
        input_text = f"{question_stem}\nchoice:\n{choices_text}"
        
        target_text = item['answerKey']
        data.append({"input": input_text, "target": target_text})

# 只保留前20个样本
data = data[:20]

# 保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
