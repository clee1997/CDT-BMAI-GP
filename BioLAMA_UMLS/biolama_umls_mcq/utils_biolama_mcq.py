def doc_to_text(doc):

    pid2prompt_meta = {'UR44': {'template': '[X] treats [Y] .'}, 'UR221': {'template': '[X] has a genetic association with [Y] .'}, 'UR45': {'template': '[X] treats [Y] .'}, 'UR48': {'template': '[X] results in [Y] .'}, 'UR211': {'template': '[X] involves [Y] .'}, 'UR214': {'template': '[Y] causes [X] .'}, 'UR256': {'template': '[Y] has a genetic association with [X] .'}, 'UR588': {'template': '[X] involves [Y] process .'}, 'UR254': {'template': '[X] has symptoms such as [Y] .'}, 'UR180': {'template': '[Y] is finding of disease [X] .'}, 'UR116': {'template': '[X] is clinically associated with [Y] .'}, 'UR625': {'template': '[X] has a genetic association with [Y] .'}, 'UR46': {'template': '[X] should not be used in the presence of [Y] disease .'}, 'UR173': {'template': '[X] is caused by [Y] .'}, 'UR49': {'template': '[X] has a mechanism of action of [Y] .'}, 'UR50': {'template': '[X] is a therapeutic class of [Y] .'}, 'UR124': {'template': 'The most widely used drug for preventing [X] is [Y] .'}}
    template =  pid2prompt_meta[doc["predicate_id"]]["template"]
    subject = doc["sub_label"]
    sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

    prefix = f'Consider the following sentence: "{sentence}"'
    mcq_suffix = "\n\nWhich noun-phrase should <BLANK> be replaced with? Choose a single correct answer from the four options given below."
    mcq_suffix += "\n\n"
        
    
    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']
    
    for i, op in enumerate(mcq_options):
        mcq_suffix += f"\t{mcq_symbols[i]}. {op}\n" 

    prompt = prefix + mcq_suffix 
    
    return f"{prompt}\n"

def doc_to_target(doc):

    return int(doc['cop'])

def doc_to_choice(doc):
    
    symbols_n_texts = []
    
    mcq_options, mcq_symbols = [doc['opa'], doc['opb'], doc['opc'], doc['opd']], ['A', 'B', 'C', 'D']
    
    for i in range(4):
    
        symbols_n_texts.append(f"{mcq_symbols[i]}. {mcq_options[i]}")
    
    return symbols_n_texts 