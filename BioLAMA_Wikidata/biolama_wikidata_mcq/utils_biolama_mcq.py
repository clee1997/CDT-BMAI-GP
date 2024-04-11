symbols_only = False

def doc_to_text(doc):

    pid2prompt_meta = {"P2176": {"template": "the standard treatment for patients with [X] is a drug such as [Y]."},
                              "P2175": {"template": "[X] has effects on diseases such as [Y]."},
                              "P4044": {"template": "[X] cures diseases such as [Y]]."},
                              "P780": {"template": "[X] has symptoms such as [Y]."},
                              "P2293": {"template": "gene [X] has a genetic association with diseases such as [Y]."}}
    
    template =  pid2prompt_meta[doc["predicate_id"]]["template"]
    subject = doc["sub_label"]
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