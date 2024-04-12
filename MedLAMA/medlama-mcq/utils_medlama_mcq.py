
symbols_only = False

def doc_to_text(doc):

    # some logic here to get which dataset (out of the 4 we're looking at) 'doc' comes from. 
    # and add appropriate template mapping here. 

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

    # Make the template for that specific doc.
    template = pid2prompt_meta[doc["rel"]]["template"]
    subject = doc["head_name"]
    sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

    prefix = f'Consider the following sentence: "{sentence}"'
    mcq_suffix = "\n\nWhich noun-phrase should <BLANK> be replaced with? Choose a single correct answer from the four options given below."
    
    # potentially add output format specification here...
    if symbols_only:
        mcq_suffix += 'Only output a single alphabet as your answer. For example, instead of "D. Renal cell carcinoma", just output "D".\n\n' # this DOES change accuracy 
    else:
        mcq_suffix += "\n\n"

    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']
    
    for i, op in enumerate(mcq_options):
        mcq_suffix += f"\t{mcq_symbols[i]}. {op}\n" 

    prompt = prefix + mcq_suffix 
    
    return f"{prompt}\n"

def doc_to_target(doc):

    # mcq_symbols = ["A", "B", "C", "D"]
    return int(doc['cop']) # expects index when there's doc_to_choice.

def doc_to_choice(doc):
    symbols_n_texts = []
    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']
    
    for i in range(4):
        # symbols_n_texts.append([mcq_symbols[i], mcq_symbols[i].lower(), f"{mcq_symbols[i]}. {mcq_options[i]}"])
        symbols_n_texts.append(f"{mcq_symbols[i]}. {mcq_options[i]}")
    return mcq_symbols if symbols_only else symbols_n_texts 
