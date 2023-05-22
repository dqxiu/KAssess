from dis import Instruction
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, AutoTokenizer, AutoModelForMaskedLM, TransfoXLTokenizer, TransfoXLLMHeadModel
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
sys.path.append(YOUR_PATH)
from LAMA.lama.eval_generation_my import *
queryInstance = WikidataQueryController()
queryInstance.init_database_connection()

import nltk
import math
from nltk.corpus import stopwords

import jsonlines


if __name__ == '__main__':
    model_name = 'gpt2-xl'
    random.seed(1)
    print("⏰ Loading data....")
    rootdir = "YOUR_PROJECT_PATH/data/my_TREx_false"
    all_paras = dict()
    false_fact_ids = []
    list_path = os.listdir(rootdir)
    for i in range(0, len(list_path)):
        path = os.path.join(rootdir, list_path[i])
        with jsonlines.open(path) as reader:
            all_paras[list_path[i].strip('.jsonl')] = []
            for obj in reader:
                false_fact_ids.append(obj['fact_id'])

    print(len(false_fact_ids))
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",'r') as load_f:
        rel2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/example.json",'r') as load_f:
        all_trex = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/symbol2text.json",
              'r') as load_f:
        rel_dict = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub_ids.json",
              'r') as load_f:
        rel2sub_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub2rate.json",
              'r') as load_f:
        rel2sub2rate = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/single_tok_objdict.json",'r') as load_f:
        single_tok_objdict = json.load(load_f)
    with open(f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_{model_name}_vocab.json",'r') as load_f:
        obj2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allsub2alias.json",'r') as load_f:
        sub2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2rel.json",'r') as load_f:
        obj2rel = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/highfreq_falseobj_forrel_renew.json",'r') as load_f:
        false_dict = json.load(load_f)
    lama_relation_list = []
    ori_relations = dict()
    with jsonlines.open('YOUR_PROJECT_PATH/data/relations.jsonl', 'r') as reader:
        for line in reader:
            lama_relation_list.append(line["relation"])
            ori_relations[line["relation"]] = line
    print("😄 All data loaded.\n")
    

    save_dict = {}
    rel2sampled_items = {}
    for fact_id in tqdm(range(len(all_trex))):
        if str(fact_id) in false_fact_ids:
            cur_fact = all_trex[fact_id]
            rel_id = cur_fact['relation']
            if rel_id not in save_dict.keys():
                save_dict[rel_id] = []
            cur_fact['fact_id'] = fact_id
            save_dict[rel_id].append(cur_fact)

    for rel_id in save_dict.keys(): 
        sampled_items = random.sample(save_dict[rel_id], min(100,len(save_dict[rel_id])))
        rel2sampled_items[rel_id] = sampled_items
        with jsonlines.open(f'YOUR_PROJECT_PATH/data/my_TREx_false_p/{rel_id}.jsonl', 'w') as writer:
            
            sampled_items = rel2sampled_items[rel_id]
            for item in sampled_items:
                writer.write(item)
