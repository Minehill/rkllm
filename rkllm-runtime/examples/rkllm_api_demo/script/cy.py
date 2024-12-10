# -*- coding: utf-8 -*-

import json

# ��ȡ JSONL �ļ�
input_file = '/userdata/repos/datasets/rkllm_code/rknn-llm/rkllm-runtime/examples/rkllm_api_demo/src/dev_rand_split.jsonl'
output_file = 'converted_data.json'

data = []
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        item = json.loads(line)
        question_stem = item['question']['stem']
        choices = item['question']['choices']
        
        # ����ѡ������
        choices_text = "\n".join([f"{choice['label']}: {choice['text']}" for choice in choices])
        
        # ���� input �ֶ�
        input_text = f"{question_stem}\nchoice:\n{choices_text}"
        
        target_text = item['answerKey']
        data.append({"input": input_text, "target": target_text})

# ֻ����ǰ20������
data = data[:20]

# ����Ϊ JSON �ļ�
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
