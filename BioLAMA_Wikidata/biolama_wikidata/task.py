import json
import os
from typing import Union, List
import ast
import collections
import re
import string
import random

from datasets.arrow_dataset import Dataset
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean


ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred): 
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def at_least_one_match_fn(gold, preds):

    gold, preds = [item.lower() for item in gold], [item.lower() for item in preds]

    hit = 0
    for pred in preds:
        if pred in gold:
            hit = 1 
            break 

    return hit

def compute_f1_from_lists(gold, preds):


    f1_scores = [0]*len(preds)
    for idx, pred in enumerate(preds):
        f1_scores[idx] += max(compute_f1(a, pred) for a in gold)

    return max(f1_scores)

def get_key_spec_strings(num_keys, synonyms=False):
    keys = [f'"top_{i}_synonyms"' for i in range(1, num_keys + 1)] if synonyms else [f'"top_{i}"' for i in range(1, num_keys + 1)]
    return ', '.join(keys[:-1]) + f' and {keys[-1]}'


class BioLAMA_Wikidata(ConfigurableTask):
    DATASET_PATH = "CDT-BMAI-GP/Biolama-Wikidata"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0 
    DATASET_NAME = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}}), 
        self.k_in_topk = 5
    
    def doc_to_text(self, doc):

        pid2prompt_meta = {"P2176": {"template": "the standard treatment for patients with [X] is a drug such as [Y]."},
                           "P2175": {"template": "[X] has effects on diseases such as [Y]."},
                           "P4044": {"template": "[X] cures diseases such as [Y]]."},
                           "P780": {"template": "[X] has symptoms such as [Y]."},
                           "P2293": {"template": "gene [X] has a genetic association with diseases such as [Y]."}}
        
        # Make the template for that specific doc.
        template =  pid2prompt_meta[doc["predicate_id"]]["template"]

        subject = doc["sub_label"]
        sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'

        prompt = prefix + suffix
        return f"{prompt}\n"

    def doc_to_target(self, doc):
        objects = []
        if 'obj_labels' in doc:
            objects = doc['obj_labels']  
        elif 'obj_label' in doc:
            objects = [doc['obj_label']]

        if 'obj_aliases' in doc:
            objects += [a for al in doc['obj_aliases'] for a in al]

        lower_objects = list(dict.fromkeys([obj.lower() for obj in objects]))

        return lower_objects

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset['validation']

    def test_docs(self) -> Dataset:
        return self.dataset['test']

    def process_results(self, doc, results): 

        gold = self.doc_to_target(doc) 

        def dict_to_list(topk_dict, k=self.k_in_topk):
            dict_vals = [([topk_dict[key]] if isinstance(topk_dict[key], str) else topk_dict[key]) for key in topk_dict.keys()]
            flattened_list = [item for sublist in dict_vals for item in sublist]
            return flattened_list
        
        def json_string_to_list(input_string):
            res = None
            try:
                res = dict_to_list(ast.literal_eval(results[0]))
            except:
                res = [""]*self.k_in_topk*3
            return res

        res = json_string_to_list(results[0])

        return {"topk_acc": at_least_one_match_fn(gold, res), "f1": compute_f1_from_lists(gold, res)} # {f"top{self.k_in_topk}_acc": topk_match_fn(gold, filtered_resps) }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {k: mean for k in ["topk_acc", "f1"]}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: True for k in ["topk_acc", "f1"]}
