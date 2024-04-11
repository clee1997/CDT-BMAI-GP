import pandas as pd
import numpy as np
import pandas as pd
import random
import json
from huggingface_hub import HfApi
import requests
import os

def get_class_instances_from_ontology(entity_id):
    """
    Args:
    - entity_id: The QID of the entity in Wikidata.
    """

    sparql_query = f"""
    SELECT ?sibling ?siblingLabel WHERE {{
    # Find the parent type/class of the example entity
    wd:{entity_id} wdt:P31 ?class.
    
    # Find other entities that are instances of the same class
    ?sibling wdt:P31 ?class.
    
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}

    LIMIT 10
    """

    # ORDER BY DESC(?popularity)

    url = 'https://query.wikidata.org/sparql'
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers, params={'query': sparql_query, 'format': 'json'})

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Failed to fetch data: Status code {response.status_code}")
        print("Response text:", response.text)
        return [], []

    siblings = set()
    for result in data["results"]["bindings"]:
        
        sibling_id = result["sibling"]["value"].split("/")[-1]
        siblings.add(sibling_id)

    return list(siblings)

def get_wikidata_entity_label(entity_id):
    """Retrieve the label of a Wikidata entity by its ID."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels",
        "languages": "en",
        "format": "json",
    }
    response = requests.get(url, params=params)
    data = response.json()
    entity = data["entities"][entity_id]
    
    try:
        label = entity["labels"]["en"]["value"]
    except:
        return ""
        
    return label

data_path = ... # 'test.jsonl'
df = pd.read_json(data_path, lines=True)

df['opa'], df['opb'], df['opc'], df['opd'] = None, None, None, None

def generate_random_cop_value():
    return random.choice(range(4))

df['cop'] = df.apply(lambda row: generate_random_cop_value(), axis=1)

def sample_negative_options(row):
    
    obj_id = row['obj_uris'][0]
    
    sibling_uris = get_class_instances_from_ontology(obj_id) 
    sibling_names = [get_wikidata_entity_label(x) for x in sibling_uris]
    sibling_names = [s for s in sibling_names if s != ""]
    
    sibling_names = set(sibling_names)
    neg_ops = list(sibling_names)
    random.shuffle(neg_ops)
    
    return neg_ops[:3] if len(neg_ops)>2 else ["NA", "NA", "NA"]

for index, row in df.iterrows():

    ops_names = ['opa', 'opb', 'opc', 'opd']
    
    df.at[index, ops_names[row['cop']]] = row['obj_labels'][0]
    
    ops_names.pop(row['cop'])
    
    neg_ops = sample_negative_options(row) 
    
    for i, col_name in enumerate(ops_names):
        df.at[index, col_name] = neg_ops[i][0]
        
df.to_csv('wikidata_test_ontology_mcq.csv', index=False)