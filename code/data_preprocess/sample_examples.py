from dis import Instruction
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from databaseconnection import *
from wikidata_get import *
import random
from statistics import mean

if __name__ == '__main__':
    save_dict = dict()
    print("⏰ Loading data....")
    with open(
            "YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",
            'r') as load_f:
        rel2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/example.json",
              'r') as load_f:
        all_trex = json.load(load_f)
    print("😄 All data loaded.\n")

    rel2sub_ids = dict()
    for rel in rel2example_ids.keys():
        if rel not in rel2sub_ids:
            rel2sub_ids[rel] = list()
        for example_id in rel2example_ids[rel]:
            rel2sub_ids[rel].append(all_trex[example_id]["subj_label"])
    for rel in rel2sub_ids.keys():
        rel2sub_ids[rel] = list(set(rel2sub_ids[rel]))
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub_ids.json", 'w') as write_f:
        json.dump(rel2sub_ids, write_f, indent=4, ensure_ascii=False)