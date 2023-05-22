'''
Generating high-frequency false answers for each relation
'''
import json
import torch
import jsonlines
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, AutoTokenizer, AutoModelForMaskedLM, TransfoXLTokenizer, TransfoXLLMHeadModel
model_name = 'gpt2'
# model_name = 'gpt2-xl'
# model_name = 'openai-gpt'
device = 'cuda'
import nltk
import math
from nltk.corpus import stopwords


def highfreq_false_single_token(model, tokenizer, device, rel_id, templates, stopwords_ids=None, all_possible_objalias=None, candis=None, obj2alias=None):

    cur_possible_answers = dict()
    for obj_id in all_possible_objalias[rel_id]:
        possible_aliases = obj2alias[obj_id]
        for alias in possible_aliases:
            if alias not in cur_possible_answers:
                cur_possible_answers[alias] = obj_id
    temp_flag = 0
    for temp_id in range(len(templates)):
        prompt_text = templates[temp_id]
        encodings = tokenizer(prompt_text,
                                return_tensors='pt').to(device)
        input_ids = encodings.input_ids.to(device)
        
        if isinstance(model,GPT2LMHeadModel):
            outputs = model(input_ids)
            probs = torch.nn.functional.softmax(outputs.logits[0,-1],dim=-1)
        elif isinstance(model,TransfoXLLMHeadModel):
            outputs = model(**encodings, mems=None)['prediction_scores']
            probs = torch.nn.functional.softmax(outputs[0,-1],dim=-1)
        if temp_flag == 0:
            prob_sum = probs
            temp_flag = 1
        else:
            prob_sum = torch.add(prob_sum, probs)
            
    top5_prob = 0
    top5_id = 0
    top10000probs, top10000ids = torch.topk(prob_sum, 10000, dim=0, largest=True, sorted=True)
    
    vocab_size = probs.shape[0]
    top5_predictions = []
    for token_id in top10000ids:
        if len(top5_predictions) == 5:
            break
        if int(token_id) in stopwords_ids:
            continue
        prediction = tokenizer.convert_ids_to_tokens([token_id])[0].replace('Ġ', '')

        label = '1'
        # manual check added
        if prediction not in cur_possible_answers:
            # continue
            label = '0'
        
        top5_predictions.append({prediction: label})
    return top5_predictions

if __name__ == "__main__":
    if 'gpt' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    else:
        tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103").to(device)

    stopwords = stopwords.words('english')
    stopwords.extend(['I', 'J', 'K', 'without','.','"',',' ,'[', ']', 'The'])
    stopwords_ids = [tokenizer.encode(' '+ stopword)[0] for stopword in stopwords]
    
    with open("YOUR_PROJECT_PATHsymbol2text.json",
              'r') as load_f:
        rel_dict = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allobj2alias.json",'r') as load_f:
        obj2alias = json.load(load_f)
        new_obj2alias = dict()
    all_possible_objalias = dict()
    lama_relation_dict = dict()
    with jsonlines.open('YOUR_PROJECT_PATHLAMA/data/relations.jsonl', 'r') as reader:
        for line in reader:
            lama_relation_dict[line["relation"]] = line["template"]  
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2rel.json",'r') as load_f:
        obj2rel = json.load(load_f)
        for obj_id in obj2rel.keys():
            for rel_id in obj2rel[obj_id]:
                if rel_id not in all_possible_objalias:
                    all_possible_objalias[rel_id] = []
                all_possible_objalias[rel_id].append(obj_id)
                
    valid_rel_num = 0
    save_dict = dict()
    for rel_id in rel_dict.keys():
        if rel_id in lama_relation_dict.keys(): 
            template = lama_relation_dict[rel_id] 
            if template[-5:] == '[Y] .':
                template = template.replace('[X]', '[X]').replace('[Y]', '').replace('.', '')
                top5_prediction = highfreq_false_single_token(model, tokenizer, device, rel_id, [template], stopwords_ids=stopwords_ids, all_possible_objalias=all_possible_objalias, obj2alias=obj2alias)
                save_dict[rel_id] = {"top5": top5_prediction, "templates": template}
    
    with open(f"YOUR_PROJECT_PATHdata/cleaned_T_REx/highfreq_falseobj_forrel_renew_gpt2.json", 'w') as write_f:
	    json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
