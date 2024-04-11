symbols_only = False


def doc_to_text(doc):
    # some logic here to get which dataset (out of the 4 we're looking at) 'doc' comes from.
    # and add appropriate template mapping here.
    pid2prompt_meta = {
        'CD1': {'template': '[X] prevents diseases such as [Y].'},
        'CD2': {'template': '[X] exposure is associated with significant increases in diseases such as [Y].'},
        'CG1': {'template': '[X] treatment decreases the levels of [Y] expression.'},
        'CG17': {'template': '[X] treatment increases the levels of [Y] expression.'},
        'CG18': {'template': '[X] upregulates [Y] protein.'},
        'CG2': {'template': '[X] results in decreased activity of [Y] protein.'},
        'CG21': {'template': '[X] results in increased phosphorylation of [Y] protein.'},
        'CG4': {'template': '[X] results in increased activity of [Y] protein.'},
        'CG6': {'template': '[X] treatment decreases the levels of [Y] expression.'},
        'CG9': {'template': '[X] binds to [Y] protein.'},
        'CP1': {'template': '[X] analog results in decreased phenotypes such as [Y] .'},
        'CP2': {'template': '[X] induces phenotypes such as [Y].'},
        'CP3': {'template': '[X] affects phenotypes such as [Y].'},
        'GD1': {'template': 'Gene [X] is associated with diseases such as [Y] .'},
        'GP1': {'template': 'Gene [X] is associated with pathways such as [Y].'}}

    template = pid2prompt_meta[doc["predicate_id"]]["template"]
    subject = doc["sub_label"]
    sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

    prefix = f'Consider the following sentence: "{sentence}"'
    mcq_suffix = "\n\nWhich noun-phrase should <BLANK> be replaced with? Choose a single correct answer from the four options given below."

    # potentially add output format specification here...
    if symbols_only:
        mcq_suffix += 'Only output a single alphabet as your answer. For example, instead of "D. Renal cell carcinoma", just output "D".\n\n'  # this DOES change accuracy
    else:
        mcq_suffix += "\n\n"

    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']

    for i, op in enumerate(mcq_options):
        mcq_suffix += f"\t{mcq_symbols[i]}. {op}\n"

    prompt = prefix + mcq_suffix

    return f"{prompt}\n"


def doc_to_target(doc):
    # mcq_symbols = ["A", "B", "C", "D"]
    return int(doc['cop'])  # expects index when there's doc_to_choice.


def doc_to_choice(doc):
    symbols_n_texts = []

    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']

    for i in range(4):
        # symbols_n_texts.append([mcq_symbols[i], mcq_symbols[i].lower(), f"{mcq_symbols[i]}. {mcq_options[i]}"])
        symbols_n_texts.append(f"{mcq_symbols[i]}. {mcq_options[i]}")

    return mcq_symbols if symbols_only else symbols_n_texts

# /lm-evaluation-harness/lm_eval/api/task.py

# elif self.OUTPUT_TYPE == "generate_until":
#     gold = self.doc_to_target(doc)
#     result = results[0]
#     if self.config.doc_to_choice is not None:
#         # If you set doc_to_choice,
#         # it assumes that doc_to_target returns a number.
#         choices = self.doc_to_choice(doc)
#         gold = choices[gold]
#     # we expect multiple_targets to be a list.
#     elif self.multiple_target:
#         gold = list(gold)
#     elif type(gold) != type(result):
#         # cast gold to the same type as result
#         gold = type(result)(gold)

#     for metric in self._metric_fn_list.keys():
#         if self.multiple_target:
#             # in the case where we have multiple targets,
#             # return true if any are true
#             # TODO: this may break for multipLe_target, non zero-or-1 metrics
#             scores = []
#             if not isinstance(gold, list):
#                 # sometimes, a multiple_target dataset has exceptions where one doc has only one string answer
#                 # print(gold)
#                 gold = [gold]
#             if metric == "exact_match":
#                 result = [result for _ in range(len(gold))]
#                 scores = self._metric_fn_list[metric](
#                     references=gold,
#                     predictions=result,
#                     **self._metric_fn_kwargs[metric],
#                 )[metric]
#                 result_score = 1.0 if scores > 0.0 else 0.0
#             else:
#                 for gold_option in gold:
#                     try:
#                         result_score = self._metric_fn_list[metric](
#                             references=[gold_option],
#                             predictions=[result],
#                             **self._metric_fn_kwargs[metric],
#                         )
#                     except (
#                         TypeError
#                     ):  # TODO: this is hacky and I don't want to do it
#                         result_score = self._metric_fn_list[metric](
#                             [gold_option, result]
#                         )
#                     if isinstance(result_score, dict):
#                         # TODO: this handles the case where HF evaluate returns a dict.
#                         result_score = result_score[metric]
#                     scores.append(result_score)
#                 if any(scores):
#                     result_score = 1.0
#                 else:
#                     result_score = 0.0
#         else:
#             try:
#                 result_score = self._metric_fn_list[metric](
#                     references=[gold],
#                     predictions=[result],
#                     **self._metric_fn_kwargs[metric],
#                 )
#             except TypeError:  # needed for now in order to use a different interface between our own metrics and HF Evaluate metrics
#                 result_score = self._metric_fn_list[metric]([gold, result])
#             if isinstance(result_score, dict):
#                 # TODO: this handles the case where HF evaluate returns a dict.
#                 result_score = result_score[metric]
#         result_dict[metric] = result_score
# else:
#     raise ValueError(
#         f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
#         "'loglikelihood', 'loglikelihood_rolling', 'generate_until' or 'multiple_choice'",
#     )