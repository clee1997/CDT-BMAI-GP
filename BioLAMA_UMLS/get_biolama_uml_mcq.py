import pandas as pd
import numpy as np
import pandas as pd
import random

import json
from huggingface_hub import HfApi

import os

# hf_path = 'JadeCheng/Biolama-umls'
# download_path = 

# API = HfApi()
# API.snapshot_download(repo_id=hf_path, local_dir=download_path, repo_type='dataset')


dev_set_path = ... #
df_val = pd.read_json(dev_set_path, lines=True)

df_val['opa'], df_val['opb'], df_val['opc'], df_val['opd'] = None, None, None, None

def generate_random_cop_value():
    return random.choice(range(4))

df_val['cop'] = df_val.apply(lambda row: generate_random_cop_value(), axis=1)


def split_df_by_column_values(df, column_name):

    unique_values = df[column_name].unique()

    return {value: df[df[column_name] == value] for value in unique_values}

df_val_by_reltype = split_df_by_column_values(df_val, 'predicate_id')

def sample_negative_options(row):
    id_ = row['predicate_id']
    
    sampled_df = df_val_by_reltype[id_].sample(n=4)
    
    if row['uuid'] in sampled_df['uuid'].values:
        sampled_df = sampled_df[sampled_df['uuid'] != row['uuid']]
    else:
        sampled_df = sampled_df.iloc[:3]
    
    neg_ops = sampled_df['obj_labels'].tolist()
    random.shuffle(neg_ops)
    
    return neg_ops


for index, row in df_val.iterrows():
    ops_names = ['opa', 'opb', 'opc', 'opd']
    
    df_val.at[index, ops_names[row['cop']]] = row['obj_labels'][0]
    
    ops_names.pop(row['cop'])
    
    neg_ops = sample_negative_options(row)
    
    for i, col_name in enumerate(ops_names):
        df_val.at[index, col_name] = neg_ops[i][0]
        

df_val.to_csv('processed_umls_test.csv', index=False)