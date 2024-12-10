# -*- coding: utf-8 -*-
import pandas as pd

# 读取 .parquet 文件
df = pd.read_parquet('/userdata/repos/datasets/gsm8k/main/train-00000-of-00001.parquet')

# 将 DataFrame 转换为 .jsonl 格式并保存
df.to_json('train_main.jsonl', orient='records', lines=True)