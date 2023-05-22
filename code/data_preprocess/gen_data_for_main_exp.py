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
    random.seed(1)
    print("⏰ Loading data....")
    rootdir = YOUR_PATH + "/data/pararel_data"
    all_paras = dict()
    list_path = os.listdir(rootdir)
    for i in range(0, len(list_path)):
        path = os.path.join(rootdir, list_path[i])
        with jsonlines.open(path) as reader:
            all_paras[list_path[i].strip('.jsonl')] = []
            for obj in reader:
                all_paras[list_path[i].strip('.jsonl')].append(obj)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",'r') as load_f:
        rel2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHrelation2template.json",
              'r') as load_f:
        rel2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/example.json",'r') as load_f:
        all_trex = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub_ids.json", 'r') as load_f:
        rel2sub_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub2rate.json", 'r') as load_f:
        rel2sub2rate = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/single_tok_objdict.json",'r') as load_f:
        single_tok_objdict = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allsub2alias.json",'r') as load_f:
        sub2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2rel.json",'r') as load_f:
        obj2rel = json.load(load_f)
    with open("$YOUR_PROJECT_PATHdata/cleaned_T_REx/highfreq_falseobj_forrel_renew.json",'r') as load_f:
        false_dict = json.load(load_f)
    print("😄 All data loaded.\n")
    
    save_dict = {}
    save_false_dict = {'rel_ids':[]}
    for fact_id in tqdm(range(len(all_trex))):
        cur_fact = all_trex[fact_id]
        cur_fact['fact_id'] = str(fact_id)
        rel_id = cur_fact['relation']
        if rel_id not in save_dict.keys():
            save_dict[rel_id] = []
        save_dict[rel_id].append(cur_fact)

    print(f"rel2alias key num: {len(rel2alias.keys())}")
    rel2sampled_items = dict() 
    for rel_id in save_dict.keys(): 
        if rel_id not in rel2alias.keys():
            continue
        with jsonlines.open(f'YOUR_PROJECT_PATHLAMA/data/my_TREx_main_new/{rel_id}.jsonl', 'w') as writer:
            sampled_items = random.sample(save_dict[rel_id], min(20,len(save_dict[rel_id])))
            rel2sampled_items[rel_id] = sampled_items
            for item in sampled_items:
                writer.write(item)

    print(f"num of relations: {len(save_dict.keys())}")
    print(f"num of sampled facts: {sum([len(rel2sampled_items[rel_id]) for rel_id in rel2sampled_items.keys()])}")
    print(f"num of sampled relations: {len(rel2sampled_items.keys())}")