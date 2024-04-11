from transformers import AutoTokenizer, AutoModel
import torch

model_id = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, output_hidden_states=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    model.cuda()


def get_embeddings(text, token_length):
    tokens = tokenizer(text, max_length=token_length, padding='max_length', truncation=True)
    input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask).hidden_states[-1]
    return torch.mean(output, dim=1).cpu().numpy()


def calculate_similarity(out1, out2):
    sim = cosine_similarity(out1, out2)[0][0]
    return sim
    # print('Similarity:', sim)


import pandas as pd
import json
# import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random


# ctd_paths = ['./ctd/test.jsonl', './ctd/dev.jsonl', './ctd/train.jsonl']
# umls_paths = ['./umls/test.jsonl', './umls/dev.jsonl', './umls/train.jsonl']
wikidata_paths = ['./wikidata/test.jsonl', './wikidata/dev.jsonl', './wikidata/train.jsonl']

# output_filepaths = ['./ctd/test_updated.jsonl', './ctd/dev_updated.jsonl', './ctd/train_updated.jsonl']
# output_filepaths = ['./umls/test_updated.jsonl', './umls/dev_updated.jsonl', './umls/train_updated.jsonl']
output_filepaths = ['./wikidata/test_updated.jsonl', './wikidata/dev_updated.jsonl', './wikidata/train_updated.jsonl']

for i, path in enumerate(wikidata_paths):
    data = [json.loads(line) for line in open(path, 'r')]
    df = pd.DataFrame(data)

    print("finish to read file")


    obj_label_vectors = {label.lower(): get_embeddings(label.lower(),len(label)) for label in pd.unique(df['obj_labels'].explode())}
    #
    print("finish getting embeddings")

    df['obj_label_vec'] = df['obj_labels'].apply(
        lambda labels: obj_label_vectors[labels[0].lower()] if labels else np.zeros(300))

    # df.to_csv('./test.csv', index=False)
    def find_similar(current_index, grouped_df, max_val):
        current_vector = grouped_df.iloc[current_index]['obj_label_vec']
        current_row = grouped_df.iloc[current_index]
        current_labels = current_row['obj_labels']

        candidates = grouped_df.drop(index=current_index)

        if current_labels:
            candidates = candidates[~candidates['obj_labels'].apply(lambda labels: labels[0] == current_labels[0])]

        candidates_filtered = candidates.copy()
        candidates_filtered['temp_first_label'] = candidates_filtered['obj_labels'].apply(lambda x: x[0] if x else None)
        candidates_filtered = candidates_filtered.drop_duplicates(subset=['temp_first_label'])

        candidates_filtered['similarity'] = candidates_filtered['obj_label_vec'].apply(
            lambda x: calculate_similarity(current_vector, x))

        candidates_filtered = candidates_filtered[(candidates_filtered['similarity'] < max_val)]

        top_3_similar = candidates_filtered.sort_values(by='similarity', ascending=False).head(3)

        top_3_similar.drop(columns=['temp_first_label', 'similarity'], inplace=True)

        return top_3_similar



    df['opa'], df['opb'], df['opc'], df['opd'], df['cop'] = None, None, None, None, None

    grouped = df.groupby('predicate_id')

    updates = []
    for name, group in grouped:
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            non_similar_rows = find_similar(index, group, 0.90)
            sampled_labels = non_similar_rows['obj_labels'].apply(lambda x: x[0] if x else None).tolist()

            current_label = row['obj_labels'][0] if row['obj_labels'] else None

            labels_to_shuffle = sampled_labels + [current_label]

            random.shuffle(labels_to_shuffle)
            cop_index = labels_to_shuffle.index(current_label)
            updates.append((row['uuid'], *labels_to_shuffle, cop_index))#:[-1]), labels_to_shuffle[-1]))

    print("finish updating")
    # print(updates)
    for uuid, opa, opb, opc, opd, cop in updates:
        df.loc[df['uuid'] == uuid, ['opa', 'opb', 'opc', 'opd', 'cop']] = [opa, opb, opc, opd, cop]

    columns_to_include = [col for col in df.columns if col != 'obj_label_vec']


    # output_filepath = './train_version2.jsonl'
    df[columns_to_include].to_json(output_filepaths[i], orient='records', lines=True)
    print("finish all")