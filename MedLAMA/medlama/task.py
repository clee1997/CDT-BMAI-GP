import ast
import re
import string
import collections
from lm_eval.api.task import ConfigurableTask
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


def at_least_one_match_fn(gold, preds):
    gold, preds = [item.lower() for item in gold], [item.lower() for item in preds]
    hit = 0
    for pred in preds:
        if pred in gold:
            hit = 1
            break
    return hit


def compute_f1(a_gold, a_pred):  # inputs are strings
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


def compute_f1_from_lists(gold, preds):
    f1_scores = [0] * len(preds)
    for idx, pred in enumerate(preds):
        f1_scores[idx] += max(compute_f1(a, pred) for a in gold)
    return max(f1_scores)


class MedLAMA(ConfigurableTask):
    DATASET_PATH = "CDT-BMAI-GP/MedLAMA"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0  # "Yaml"
    DATASET_NAME = None

    # CONFIG = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})
        self.unfiltered = None
        self.prediction = None
        self.hit = 0
        self.k_in_topk = 5

    def doc_to_text(self, doc):

        # manual prompt template (MedLAMA)
        pid2prompt_meta = {
            'associated_morphology_of': {'template': '[X] is associated morphology of [Y] .'},
            'disease_has_abnormal_cell': {'template': '[X] has the abnormal cell [Y] .'},
            'disease_has_associated_anatomic_site': {
                'template': 'The disease [X] can stem from the associated anatomic_site [Y] .'},
            'disease_has_normal_cell_origin': {'template': 'The disease [X] stems from the normal cell [Y] .'},
            'disease_has_normal_tissue_origin': {'template': 'The disease [X] stems from the normal tissue [Y] .'},
            'disease_mapped_to_gene': {'template': 'The disease [X] is mapped to gene [Y] .'},
            'disease_may_have_associated_disease': {
                'template': 'The disease [X] might have the associated disease [Y] .'},
            'disease_may_have_finding': {'template': '[X] may have [Y] .'},
            'disease_may_have_molecular_abnormality': {
                'template': 'The disease [X] may have molecular abnormality [Y] .'},
            'gene_associated_with_disease': {'template': 'The gene [X] is associatied with disease [Y] .'},
            'gene_encodes_gene_product': {'template': 'The gene [X] encodes gene product [Y] .'},
            'gene_product_encoded_by_gene': {'template': 'The gene product [X] is encoded by gene [Y] .'},
            'gene_product_has_associated_anatomy': {
                'template': 'The gene product [X] has the associated anatomy [Y] .'},
            'gene_product_has_biochemical_function': {'template': '[X] has biochemical function [Y] .'},
            'gene_product_has_chemical_classification': {'template': 'The gene product [X] is a type of [Y] .'},
            'gene_product_plays_role_in_biological_process': {
                'template': 'The gene product [X] plays role in biological process [Y] .'},
            'has_physiologic_effect': {'template': '[X] has physiologic effect of [Y] .'},
            'may_prevent': {'template': '[X] may be able to prevent [Y] .'},
            'may_treat': {'template': '[X] might treat [Y] .'},
            'occurs_after': {'template': '[X] occurs after [Y] .'}}

        template = pid2prompt_meta[doc["rel"]]["template"]
        subject = doc["head_name"]
        sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        question = '\n\n-> Which noun-phrase should <BLANK> be filled with? '
        suffix = ('Give me 5 most probable candidates. '
                  'Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", '
                  'where the value for key "top_1" is the most promising entity that would replace <BLANK>.')

        prompt = prefix + question + suffix
        return f"{prompt}\n"

    def doc_to_target(self, doc):
        tails = eval(doc['tail_names_list'])
        lower_tails = [item.lower() for item in tails]
        return lower_tails

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    # def validation_docs(self):
    #     # print(f"self.dataset = {self.dataset}")
    #     return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results):
        gold = self.doc_to_target(doc)

        def dict_to_list(topk_dict, k=self.k_in_topk):
            # return
            dict_vals = [([topk_dict[key]] if isinstance(topk_dict[key], str) else topk_dict[key]) for key in
                         topk_dict.keys()]
            flattened_list = [item for sublist in dict_vals for item in sublist]
            return flattened_list

        def json_string_to_list(input_string):
            try:
                res_list = dict_to_list(ast.literal_eval(input_string))
            except:
                res_list = [""] * self.k_in_topk
            return res_list

        res = json_string_to_list(results[0])
        return {"topk_acc": at_least_one_match_fn(gold, res), "f1": compute_f1_from_lists(gold, res)}

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
