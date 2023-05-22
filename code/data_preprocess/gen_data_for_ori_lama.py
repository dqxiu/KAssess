'''
Code for generating data for LAMA in the original paper and the test sets (varied sets for variance and false positive set for spurious correlation comparison with lama) 
To ensure a fair comparison with LAMA,  we restricted the relation classes to those included in LAMA. These test sets are only applicable to 41 high-frequency relations.
'''

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
    rootdir = "YOUR_PROJECT_PATHdata/pararel_data"
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
    save_false_dict = {'rel_ids':[]}
    for fact_id in tqdm(range(len(all_trex))):
        cur_fact = all_trex[fact_id]
        cur_fact['fact_id'] = str(fact_id)
        rel_id = cur_fact['relation']
        if rel_id not in save_dict.keys():
            save_dict[rel_id] = []
        save_dict[rel_id].append(cur_fact)
        
        if rel_id in false_dict.keys():
            gold_obj = " ".join(cur_fact['token'][cur_fact['obj_start']:cur_fact['obj_end']+1])
            false_obj_candi_dict = false_dict[rel_id]["top5"][0]
            false_obj = ""
            false_obj_candi = [key for key in false_obj_candi_dict.keys()][0]
            if false_obj_candi not in gold_obj and gold_obj not in false_obj_candi and false_obj_candi_dict[false_obj_candi]!='0':
                print(false_obj_candi)
                false_obj = false_obj_candi
            if rel_id in lama_relation_list and false_obj!="":
                cur_false_fact = cur_fact.copy()
                cur_false_fact['ori_obj_label'] = " ".join(cur_fact['token'][cur_false_fact['obj_start']:cur_false_fact['obj_end']+1])
                cur_false_fact['sub_label'] = " ".join(cur_fact['token'][cur_false_fact['subj_start']:cur_false_fact['subj_end']+1])
                cur_false_fact['obj_label'] = false_obj
                save_false_dict[str(fact_id)] = cur_false_fact
                save_false_dict['rel_ids'].append(rel_id)

    with jsonlines.open(f'YOUR_PROJECT_PATH/data/relations1.jsonl', 'w') as writer1:
        with jsonlines.open(f'YOUR_PROJECT_PATH/data/relations2.jsonl', 'w') as writer2:
            with jsonlines.open(f'YOUR_PROJECT_PATH/data/relations3.jsonl', 'w') as writer3:
                    valid_para_relations = []
                    para_relations1 = []
                    para_relations2 = []
                    para_relations3 = []
                    for rel_id in all_paras.keys():
                        rel_dicts = []
                        for rel_dict_temp in all_paras[rel_id]:
                            if rel_dict_temp['pattern'][-4:] == '[Y].':
                                rel_dicts.append(rel_dict_temp)
                        if len(rel_dicts) > 4:
                            sampled_rel_dicts = random.sample(rel_dicts[1:], 3)
                            ori_relations[rel_id]['template'] = sampled_rel_dicts[0]['pattern']
                            writer1.write(ori_relations[rel_id])
                            ori_relations[rel_id]['template'] = sampled_rel_dicts[1]['pattern']
                            writer2.write(ori_relations[rel_id])
                            ori_relations[rel_id]['template'] = sampled_rel_dicts[2]['pattern']
                            writer3.write(ori_relations[rel_id])
                            valid_para_relations.append(rel_id)


    rel2sampled_items = dict() 
    for rel_id in save_dict.keys(): 
        with jsonlines.open(f'YOUR_PROJECT_PATH/data/my_TREx/{rel_id}.jsonl', 'w') as writer:
            sampled_items = random.sample(save_dict[rel_id], min(100,len(save_dict[rel_id])))
            rel2sampled_items[rel_id] = sampled_items
            for item in sampled_items:
                writer.write(item)

    for rel_id in save_dict.keys(): 
        if rel_id in all_paras.keys() and rel_id in valid_para_relations:
            with jsonlines.open(f'YOUR_PROJECT_PATH/data/my_TREx_para/{rel_id}.jsonl', 'w') as writer:
                sampled_items = rel2sampled_items[rel_id]
                for item in sampled_items:
                    writer.write(item)
                    
    total_false = 0 
    for rel_id in save_false_dict['rel_ids']: 
        if rel_id == 'rel_ids':
            continue
        if rel_id in lama_relation_list:
            with jsonlines.open(f'YOUR_PROJECT_PATH/data/my_TREx_false/{rel_id}.jsonl', 'w') as writer:
                sampled_items = rel2sampled_items[rel_id]
                sampled_items_false = []
                for sampled_item in sampled_items:
                    if str(sampled_item['fact_id']) in save_false_dict.keys():
                        sampled_items_false.append(save_false_dict[str(sampled_item['fact_id'])])
                total_false += len(sampled_items_false)
                for item in sampled_items_false:
                    writer.write(item)
    print(f"num of relations: {len(save_dict.keys())}")
    print(f"num of facts: {len(all_trex)}")
    print(f"num of facts with false obj: {total_false}")