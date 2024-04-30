import requests
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('UMLS_API_KEY')

relations = ['parents', 'children', 'ancestors', 'descendants', 'relations']

def get_all_relations(cui): 
    '''
    Retrieve all relationships associated with a known CUI
    NLM does not assert parent or child relationships between concepts.
    '''
    cui = str(cui)
    
    query = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/relations" # relations?sabs=MTH -> Retrieves NLM-asserted relationships of the CUI	
    params = {'apiKey': api_key}
    
    output = requests.get(query, params=params)
    output.encoding = 'utf-8'
    print(output.url)
        
    outputJson = output.json()
    return outputJson


# def crosswalk(cui, source='MTH', target='RXNORM'): # MTH? SRC?
        
#     cui = str(cui)
    
#     crosswalk_endpoint=f'https://uts-ws.nlm.nih.gov/rest/crosswalk/current/source/{source}'
#     path =  crosswalk_endpoint + '/' + cui
#     query = {'targetSource': target, 'apiKey': api_key}
    
#     output = requests.get(path, params = query)
#     output.encoding = 'utf-8'
#     print(output.url)
        
#     outputJson = output.json()
#     return outputJson

def get_related_entities(source_specific_id, source="RXNORM", relation='parents'): # https://documentation.uts.nlm.nih.gov/rest/relations/index.html
    '''
    Retrieves all {'parents', 'children', 'ancestors', 'descendants' or 'relations'} of a source-asserted identifier
    '''

    source_specific_id = str(source_specific_id)
    relations = ['parents', 'children', 'ancestors', 'descendants', 'relations']
    assert relation in relations, "relation has to be one of the following: 'parents', 'children', 'ancestors', 'descendants' and 'relations'. "
    
    query = f"https://uts-ws.nlm.nih.gov/rest/content/current/source/{source}/{source_specific_id}/{relation}"
    params = {'apiKey': api_key}

    output = requests.get(query, params=params)
    output.encoding = 'utf-8'
    print(output.url)
        
    outputJson = output.json()
    return outputJson


# https://uts.nlm.nih.gov/uts/umls/concept/C0139919

def simple_url_request(url):
    
    query = {'apiKey': api_key}
    output = requests.get(url, params = query)
    output.encoding = 'utf-8'
    print(output.url)
    
    outputJson = output.json()
    return outputJson
    