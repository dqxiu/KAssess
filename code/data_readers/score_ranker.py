import json
from statistics import geometric_mean
import numpy as np
import pandas as pd
def topk_rels(rel2birr, k, mode):
    if mode=='best':
        sorted_rels = sorted(rel2birr.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_rels = sorted(rel2birr.items(), key=lambda x: x[1], reverse=False)

    topk_rels = sorted_rels[:k]
    return topk_rels
mode = 'bs_rr_sub_rel_main'

if 'bs' in mode:
    thresh = 22

    path_name = YOUR_SCORE_PATH
    with open(path_name,'r') as load_f:
        load_dict = json.load(load_f)
    if 'bs' in mode:
        sub_replace_result_name = 'rr_bs_sub_result'
        rel_replace_result_name = 'rr_bs_rel_result'
    rr_sub_label = [] 
    rr_rel_label = []
    birr = []
    rel2birr = {}
    valid_num = 0
    for fact_id in load_dict:
        rel_id = load_dict[fact_id]['rel_id']
        if 'sub' in mode:
            rr_sub_result = load_dict[fact_id][sub_replace_result_name]
            if float(rr_sub_result['score']) > thresh:
                rr_sub_label.append(1)
            else:
                rr_sub_label.append(0)
        if 'rel' in mode:
            rr_rel_result = load_dict[fact_id][rel_replace_result_name]
            if float(rr_rel_result['score']) > thresh:
                rr_rel_label.append(1)
            else:
                rr_rel_label.append(0)
        if 'sub' in mode and 'rel' in mode:
            if float(rr_sub_result['score'])==0.0:
                rr_sub_result['score'] = 0.000001
            if float(rr_rel_result['score'])==0.0:
                rr_rel_result['score'] = 0.000001
            cur_birr = geometric_mean([float(rr_sub_result['score']),float(rr_rel_result['score'])])
            if cur_birr > thresh:
                birr.append(1)
                cur_birr_label = 1
            else:
                birr.append(0)
                cur_birr_label = 0
                
            if rel_id not in rel2birr:
                rel2birr[rel_id] = [] 
            rel2birr[rel_id].append(cur_birr_label)

            valid_num += 1

    if 'sub' in mode:
        print(f"For sub replacement, the valid number is {valid_num}, the true positive is {sum(rr_sub_label)/valid_num}")
    if 'rel' in mode:
        print(f"For rel replacement, the valid number is {valid_num}, the true positive is {sum(rr_rel_label)/valid_num}")
    if 'sub' in mode and 'rel' in mode:
        try: 
            print(f"For both replacement, the valid number is {valid_num}, the true positive is {sum(birr)/valid_num}")
        except:
            print(f"For both replacement, the valid number is {valid_num}, the true positive is {sum(birr)/valid_num}")

