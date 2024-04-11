

def doc_to_text(doc):

    pid2prompt_meta = {"P2176": {"template": "The standard treatment for patients with [X] is a drug such as [Y]."},
                       "P2175": {"template": "[X] has effects on diseases such as [Y]."},
                       "P4044": {"template": "[X] cures diseases such as [Y]."},
                       "P780": {"template": "[X] has symptoms such as [Y]."},
                       "P2293": {"template": "Gene [X] has a genetic association with diseases such as [Y]."}}
    
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



