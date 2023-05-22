from dis import Instruction
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from databaseconnection import *
from wikidata_get import *
import random
from statistics import mean
from tqdm import tqdm
from sklearn import preprocessing
from transformers import pipeline, set_seed
import jsonlines
import os
import sys
sys.path.append('YOUR_PROJECT_PATH')
from LAMA.lama.eval_generation_my import *
queryInstance = WikidataQueryController()
queryInstance.init_database_connection()
# no alias
device = 'cuda'
model_name = 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

obj_path = "YOUR_PROJECT_PATHdata/cleaned_T_REx/allobj2alias.json"
with open(obj_path,'r') as load_f:
    load_dict = json.load(load_f)
single_tok_objdict = dict()
for obj in tqdm(load_dict.keys()):
    for obj_alias in load_dict[obj]:
        obj_ids = tokenizer.encode(' ' + obj_alias.strip(), add_special_tokens=False)
        if len(obj_ids) > 1:
            continue
        if str(obj_ids[0]) not in single_tok_objdict:
            single_tok_objdict[str(obj_ids[0])] = list()
        single_tok_objdict[str(obj_ids[0])].append(obj)
    
with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/single_tok_objdict.json", 'w') as write_f:
	json.dump(single_tok_objdict, write_f, indent=4, ensure_ascii=False)


