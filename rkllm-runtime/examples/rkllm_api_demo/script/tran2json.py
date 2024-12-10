# -*- coding: utf-8 -*-
import pandas as pd

# ��ȡ .parquet �ļ�
df = pd.read_parquet('/userdata/repos/datasets/gsm8k/main/train-00000-of-00001.parquet')

# �� DataFrame ת��Ϊ .jsonl ��ʽ������
df.to_json('train_main.jsonl', orient='records', lines=True)