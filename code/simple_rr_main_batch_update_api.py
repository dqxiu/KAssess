'''
Test on the OPT-175B through API, the api is constucted following https://github.com/facebookresearch/metaseq
'''
from dis import Instruction
# import torch
import json
import openai
from transformers import AutoTokenizer, GPT2Tokenizer
import random
import torch
from statistics import mean
from tqdm import tqdm
from sklearn import preprocessing
import jsonlines
import os
import sys
import requests
sys.path.append('./')
import argparse
import nltk
import math
from nltk.corpus import stopwords
from multiprocessing.util import Finalize
import multiprocessing
from multiprocessing import  Process
random.seed(1)

headers ={
    "Content-Type": "application/json; charset=UTF-8"
    }
url = "http://10.140.24.105:6010/completions" # Replace with your API endpoint

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def data_filter(probing_facts, rel2alias, sub2alias, obj2alias, exp_mode):
    filtered_facts = dict()
    print("filtering data")
    for fact_id in probing_facts.keys():
        fact = probing_facts[fact_id]
        sub_id = fact[0]
        rel_id = fact[1]
        obj_id = fact[2]

        sub_gold_info = get_entity_defaut_alias(sub_id, sub2alias)
        obj_gold_info = get_entity_defaut_alias(obj_id, obj2alias)
        if sub_gold_info == None or obj_gold_info == None:
            print("no defaut alias for sub or obj")
            continue
        sub_gold = sub_gold_info
        obj_gold = obj_gold_info
        if rel_id not in rel2alias.keys():
            print("no single text rel")
            continue
        rel_gold_list = rel2alias[rel_id]
        filtered_facts[fact_id] = [sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold]
        if sub_id not in sub2alias.keys():
            sub2alias[sub_id] = [sub_gold]
        elif sub_gold not in sub2alias[sub_id]:
            sub2alias[sub_id].append(sub_gold)
        if obj_id not in obj2alias.keys():
            obj2alias[obj_id] = [obj_gold]
        elif obj_gold not in obj2alias[obj_id]:
            obj2alias[obj_id].append(obj_gold)
    return filtered_facts, sub2alias, obj2alias

def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_entity_defaut_alias(entity_id, whole_dict):
    if entity_id in whole_dict.keys() and len(whole_dict[entity_id])>0:
        return whole_dict[entity_id][0]
    else:
        return None

def perplexity_from_token_logits(tok_probs):
    """Calculate perplexity from token logits."""
    # tok_probs = response['choices'][0]['logprobs']['token_logprobs']
    return torch.exp(- torch.sum(tok_probs, dim=0) / tok_probs.size(0))

def condition_prob_api(predict_sent, given_sent, tokenizer, model_name, device, mode="given beta"):

    if 'opt' in model_name:
        if mode == "given beta" or mode == "given alpha":
            pyload = {"prompt": predict_sent, "max_tokens": "0", "echo": "true"}
            response = json.loads(requests.post(url, data=json.dumps(pyload), headers=headers).text)
            res = [r['text'].encode("utf-8").decode("unicode_escape") for r in response['choices']]
            res =  response['choices'][0]['logprobs']['token_logprobs']
            calculate_ids = []
            text_offset = response['choices'][0]['logprobs']['text_offset']
            pos_las = predict_sent.rfind(given_sent)
            start_s = pos_las
            end_s = len(predict_sent)
            if start_s>0:
                start_s -= 1
            for tok_id in range(len(text_offset)):
                if text_offset[tok_id] in range(start_s, end_s):
                    calculate_ids.append(tok_id)
            salient_res = [res[calculate_id] for calculate_id in calculate_ids]
            salient_perplexity = perplexity_from_token_logits(torch.tensor(salient_res))
            prob = 1 / salient_perplexity.item()
            return prob
    return prob
    
def sentence_prob_api(sentence, tokenizer,  model_name, device):
    if 'opt' in model_name:
        pyload = {"prompt": sentence, "max_tokens": "0", "echo": "true"}
        response = json.loads(requests.post(url, data=json.dumps(pyload), headers=headers).text)
        res = [r['text'].encode("utf-8").decode("unicode_escape") for r in response['choices']]
        res =  response['choices'][0]['logprobs']['token_logprobs']
        sent_perplexity = perplexity_from_token_logits(torch.tensor(res))
        prob = 1 / sent_perplexity.item()
    return prob
        

def rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate,  max_samples):
    '''
    Given rel sub replacement
    score = 
    '''
    betas = []
    beta_temps = []
    gammas = []
    all_r_gammas = dict()
    alphas = sub2alias[sub_id]
    for r_alias in rel2alias[rel_id]:
        r_alias = r_alias.strip('.').strip()
        if len(r_alias) == 0:
            continue
        for alpha in alphas:
            betas.append(r_alias.replace('[X]', alpha))
        beta_temps.append(r_alias.strip('.').strip())
    
    
    for o_alias in obj2alias[obj_id]:
        for beta in betas:
            gammas.append(beta.replace('[Y]', o_alias))
            
    betas = [beta.replace('[Y]', '').strip() for beta in betas]

    p_numerator_info = dict()
    p_denominator_info = dict()
    
    p_numerator = 0
    for beta in betas:
        p_beta = sentence_prob_api(beta, tokenizer, model, device)
        p_gamma_sum = 0
        gamma_dict = dict()
        for gamma in gammas:
            if beta not in gamma:
                continue
            p_gamma = condition_prob_api(gamma, beta, tokenizer, model, device, mode="given beta")
            gamma_dict[gamma] = round(p_gamma, 6)
            p_gamma_sum += p_gamma
        if math.isnan(p_beta) or math.isnan(p_gamma_sum):
            continue
        p_numerator += p_beta * p_gamma_sum
    p_numerator_info[beta] = { 'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict, 'other': len(betas)/len(alphas)}
    
    p_denominator = 0
    subids_list = [other_sub_id for other_sub_id in rel2sub2rate[rel_id].keys()] 
    probs_list = [rel2sub2rate[rel_id][i] for i in subids_list]
    Q_weights = probs_list
    sample_k = 4
    Q_sampled_sub_idxes = random.sample(range(len(subids_list)), min(sample_k, len(subids_list)))
    Q_sampled_sub_ids = [subids_list[idx] for idx in Q_sampled_sub_idxes]
    for beta_temp in beta_temps:
        for other_sub_id in Q_sampled_sub_ids:
            for alpha in sub2alias[other_sub_id]:
                if alpha == None or beta_temp == None:
                    continue
                p_beta = sentence_prob_api(beta_temp.replace('[X]', alpha).replace('[Y]','').strip(), tokenizer, model, device)
                Q = rel2sub2rate[rel_id][other_sub_id] * (1/len(sub2alias[other_sub_id]))
                # print(f"alpha: {alpha}")
                P_m = sentence_prob_api(alpha, tokenizer, model, device)
                p_gamma_sum = 0
                gamma_dict = dict()
                for o_alias in obj2alias[obj_id]:
                    gamma = beta_temp.replace('[X]', alpha).replace('[Y]', o_alias)
                    p_gamma = condition_prob_api(gamma, beta_temp.replace('[X]', alpha).replace('[Y]','').strip(), tokenizer, model, device, mode="given beta")
                    p_gamma_sum += p_gamma
                    gamma_dict[gamma] = round(p_gamma, 6)

                p_denominator += p_beta * p_gamma_sum
    p_denominator = p_denominator / min(sample_k, len(subids_list))
    p_denominator_info[beta] = {'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
    
    if p_denominator == 0:
        return None
    rr_result = {"score": p_numerator / p_denominator, "triplet":[sub_gold,rel_gold_list[0],obj_gold], 'p_numerator_score': p_numerator, "p_numerator_info": p_numerator_info, 'p_denominator_score': p_denominator, "p_denominator_info": p_denominator_info}
    return rr_result

def rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate,  max_samples):
    '''
    Given sub, rel replacement
    '''
    betas = []
    gammas = []
    all_r_gammas = dict() #objs可以作为哪些rel的objs,同时sub也可以作为哪些rel的subs
    alphas = sub2alias[sub_id]
    for r_alias in rel2alias[rel_id]:
        r_alias = r_alias.strip('.').strip()
        if len(r_alias) == 0:
            continue
        for alpha in alphas:
            betas.append(r_alias.replace('[X]', alpha))
    
    
    for o_alias in obj2alias[obj_id]:
        for beta in betas:
            gammas.append(beta.replace('[Y]', o_alias))
            
    betas = [beta.replace('[Y]', '').strip() for beta in betas]
    
    p_numerator_info = dict()
    p_denominator_info = dict()
    
    p_numerator = 0
    for beta in betas:
        # P_M(beta)
        p_beta = sentence_prob_api(beta, tokenizer, model, device)
        p_gamma_sum = 0
        gamma_dict = dict()
        for gamma in gammas:
            # P_M(gamma|beta)
            if beta not in gamma:
                continue
            p_gamma = condition_prob_api(gamma, beta, tokenizer, model, device, mode="given beta")
            gamma_dict[gamma] = round(p_gamma, 6)
            p_gamma_sum += p_gamma
        if math.isnan(p_beta) or math.isnan(p_gamma_sum):
            continue
        p_numerator += p_beta * p_gamma_sum
    p_numerator_info[beta] = { 'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict, 'other': len(alphas)}
    
    p_denominator = 0
    relids_list = list(obj2rel2rate[obj_id].keys())
    sample_k = 4
    Q_sampled_rel_idxes = random.sample(range(len(relids_list)), min(sample_k, len(relids_list)))
    Q_sampled_rel_ids = [relids_list[idx] for idx in Q_sampled_rel_idxes]
    for alpha in alphas:
        # P_M(alpha)
        p_alpha = sentence_prob_api(alpha, tokenizer, model, device)
        p_gamma_sum = 0
        gamma_dict = dict()
        for other_rel_id in Q_sampled_rel_ids:
            # random sample from rel2alias[other_rel_id]
            beta = random.sample(rel2alias[other_rel_id], 1)[0]
        # for beta_text in all_r_gammas[alpha]:
            # P_M(gamma|beta) sum P(Obama was born in Hawaii | Obama) all possible strings
            for o_alias in obj2alias[obj_id]:
                if beta==None or alpha==None or o_alias==None:
                    continue
                gamma = beta.replace('[X]', alpha).replace('[Y]', o_alias) 
                p_gamma = condition_prob_api(gamma, alpha, tokenizer, model, device, mode="given alpha")
                p_gamma_sum += p_gamma
                gamma_dict[gamma] = round(p_gamma, 6)
        if math.isnan(p_alpha) or math.isnan(p_gamma_sum):
            continue
        p_denominator += p_alpha * p_gamma_sum 
    p_denominator = p_denominator / (min(sample_k, len(relids_list)))
    p_denominator_info[beta] = {'p_alpha': round(p_alpha, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
    
    if p_denominator == 0:
        return None
    rr_result = {"score": p_numerator / p_denominator, "triplet":[sub_gold,rel_gold_list[0],obj_gold], 'p_numerator_score': p_numerator, "p_numerator_info": p_numerator_info, 'p_denominator_score': p_denominator, "p_denominator_info": p_denominator_info}
        # print(rr_result)
    return rr_result


def fun1(exp_mode, ori_facts, index, size, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, max_samples):
    
    if f"multiprocess_main_{exp_mode}_{model_name_replaced}_{max_samples}_orig.json" in os.listdir('./scores/'):
        with open(f"./scores/multiprocess_main_{exp_mode}_{model_name_replaced}_{max_samples}_orig.json", 'r') as saved_f:
            save_dict = json.load(saved_f)
    else:
        save_dict = dict()
    facts = random_dic(ori_facts)
    for fact_id in tqdm(facts.keys()):
        if fact_id in save_dict.keys():
            print(f"fact_id {fact_id} already in save_dict")
            continue
        fact = facts[fact_id]
        sub_id = fact[0]
        rel_id = fact[1]
        obj_id = fact[2]
        sub_gold = fact[3]
        rel_gold_list = fact[4]
        obj_gold = fact[5]
        # parents_sub = fact[6]
        # parents_obj = fact[7]
        if 'openai-gpt' in exp_mode:
            parents_obj = parents_obj.lower()
        
        if 'bs' in exp_mode:
            print("**********bs************") 
            if 'sub' in exp_mode:
                rr_bs_sub_result = rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate,  max_samples)

            if 'rel' in exp_mode:
                rr_bs_rel_result = rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate,  max_samples)
            
            if rr_bs_sub_result is None or rr_bs_rel_result is None:
                continue
            
            meta_info = {"fact_id": fact_id,"triplet": rr_bs_rel_result['triplet']}
            rr_bs_sub_result.pop('triplet')
            if float(rr_bs_sub_result['score']) > 1:
                rr_bs_sub_result['rr_bs_sub_label'] = 1
            else:
                rr_bs_sub_result['rr_bs_sub_label'] = 0
            rr_bs_rel_result.pop('triplet')
            if float(rr_bs_rel_result['score']) > 1:
                rr_bs_rel_result['rr_bs_rel_label'] = 1
            else:
                rr_bs_rel_result['rr_bs_rel_label'] = 0
                
            birr = int(rr_bs_sub_result['rr_bs_sub_label']) * int(rr_bs_rel_result['rr_bs_rel_label'])
            save_dict[str(fact_id)] = {'meta_info': meta_info, 'rr_bs_sub_result': rr_bs_sub_result, 'rr_bs_rel_result': rr_bs_rel_result, 'birr': birr, 'rel_id': rel_id}

        if len(save_dict.keys()) % 20 == 0:
            with open(f"./scores/multiprocess_main_{exp_mode}_{model_name_replaced}_{max_samples}_orig.json", 'w') as write_f:
                json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
    with open(f"./scores/multiprocess_main_{exp_mode}_{model_name_replaced}_{max_samples}_orig.json", 'w') as write_f:
        json.dump(save_dict, write_f, indent=4, ensure_ascii=False) 
    return save_dict



if __name__ == '__main__':

    model = "opt"
    model_name = 'opt'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    stopwords = stopwords.words('english')
    stopwords.extend(['I', 'J', 'K', 'without'])
    device = 'cpu'
    main_parser = argparse.ArgumentParser()
    
    main_parser.add_argument('--model_name', type=str, default="opt", help='which model')
    main_parser.add_argument('--split_id', type=str, default="0", help='which model')
    main_args = main_parser.parse_args()
    model_name = main_args.model_name + '_' + str(main_args.split_id)
    model_name_replaced = model_name.replace('/', '_')
    # exp_mode = 'bs_sub_rel_false'
    exp_mode = 'bs_sub_rel_main' # txl or openai error stop flan-xl
    # model_name = 'openai-gpt'
    max_samples = 30 # 10, 20, 30, 40, 50
    all_save_dict = dict()
     
    stopwords_ids = [tokenizer.encode(' '+ stopword)[0] for stopword in stopwords]
    
    print("⏰ Loading data....")
    with open("./data/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open("./data/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open("./data/cleaned_T_REx/relation2example_ids.json",'r') as load_f:
        rel2example_ids = json.load(load_f)
    
    if 'false' not in exp_mode and 'para' not in exp_mode and 'orig' not in exp_mode and 'freq' not in exp_mode and 'main' not in exp_mode and 'human' not in exp_mode:
        with open("./data/cleaned_T_REx/example_1000test.json",'r') as load_f:
            all_trex = json.load(load_f)
    else:
        if 'freq' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_freq"
        elif 'false_p' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_false_p"
        elif 'para' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_para"
        elif 'orig' in exp_mode:
            rootdir = "./LAMA/data/my_TREx"
        elif 'human_v1' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_human_v1"
        elif 'human_v2' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_human_v2"
        elif 'false' in exp_mode:
            rootdir = "./LAMA/data/my_TREx_false"
        elif 'main' in exp_mode:
            rootdir = f"./LAMA/data/my_TREx_main_split_{main_args.split_id}"
            
        all_trex = []
        list_path = os.listdir(rootdir)
        for i in range(0, len(list_path)):
            # 构造路径
            path = os.path.join(rootdir, list_path[i])
            with jsonlines.open(path) as reader:
                for obj in reader:
                    all_trex.append(obj)
    
    with open("./data/symbol2text.json",
                'r') as load_f:
        rel_dict = json.load(load_f)
    with open("./data/cleaned_T_REx/rel2sub_ids.json",
                'r') as load_f:
        rel2sub_ids = json.load(load_f)
    with open("./data/cleaned_T_REx/rel2sub2rate.json",
                'r') as load_f:
        rel2sub2rate = json.load(load_f)
    with open("./data/cleaned_T_REx/single_tok_objdict.json",'r') as load_f:
        single_tok_objdict = json.load(load_f)
    if 'gpt2' or 'opt' in model_name_replaced:
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_gpt2_vocab.json"
    elif 't5' in model_name_replaced:
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_t5-large_vocab.json"
    elif 'bloom' in model_name_replaced:
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_bigscience_bloom-560m_vocab.json"
    elif 'gpt3' in model_name_replaced:
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_facebook_opt-125m_vocab.json"
    elif 'llama' in model_name_replaced or 'alpaca' in model_name_replaced or 'vicuna' in model_name_replaced:
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_decapoda-research_llama-7b-hf_vocab.json"
    else: 
        vocab_path = f"./data/cleaned_T_REx/obj2alias_for_{model_name_replaced}_vocab.json"

    with open(vocab_path,'r') as load_f:
        obj2alias = json.load(load_f)
        if 'openai-opt' in model_name:
            for obj_id in obj2alias.keys():
                cur_obj_aliases = obj2alias[obj_id]
                for obj_alias_id in range(len(cur_obj_aliases)):
                    cur_obj_aliases[obj_alias_id] = cur_obj_aliases[obj_alias_id].lower()
    with open("./data/cleaned_T_REx/allsub2alias.json",'r') as load_f:
        sub2alias = json.load(load_f)
    with open("./data/cleaned_T_REx/obj2rel2rate.json",'r') as load_f:
        obj2rel2rate = json.load(load_f)
    print("😄 All data loaded.\n")
    
    # rel alias filter
    with open("./relation2template.json",
                'r') as load_f:
        rel2alias = json.load(load_f)

    para_alias = dict()
    if 'para' in exp_mode:
        para_name = exp_mode[-1]
        if para_name == 'a':
            para_name = ''
        if para_name != '':
            with jsonlines.open(f'./LAMA/data/relations{para_name}.jsonl', 'r') as reader:
                for line in reader:
                    line_rel =  line["relation"]
                    # randomly replace one alias in rel2alias[line_rel] by line["template"] 
                    alias_list = rel2alias[line_rel]
                    replaced_alias = alias_list[random.randint(0, len(alias_list)-1)]
                    alias_list.remove(replaced_alias)
                    alias_list.append(line["template"])
                    rel2alias[line_rel] = alias_list
                
    # sample_facts_num = 10
    sample_facts_num = len(all_trex) 
    sample_trex_items_ids = random.sample(range(len(all_trex)), sample_facts_num)
    sample_trex_items = dict()
    for sample_id in sample_trex_items_ids:
        if len(sample_trex_items.keys()) > sample_facts_num:
            break
        fact_id = all_trex[sample_id]['fact_id']
        item = all_trex[sample_id]
        if 'sub_label' in item.keys():
            # for false facts
            false_trans = {'Atlanta': 'Q23556', 'English': 'Q1860', 'Chicago': 'Q1297', 'France': 'Q142', 'London': 'Q84', 'French': 'Q150', 'medicine': 'Q11190', 'United': 'Q30', 'NBC': 'Q13974', 'RCA': 'Q50074604', 'ABC': 'Q169889', 'England': 'Q21', 'science': 'Q336'}
            if '#' in false_trans[item['obj_label']] or '#' in item['subj_label']:
                print("the fact is not in the original T_REx dataset")
                continue
            sample_trex_items[str(fact_id)] = (item['subj_label'], item['relation'], false_trans[item['obj_label']])
        else:
            if '#' in item['obj_label'] or '#' in item['subj_label']:
                print("the fact is not in the original T_REx dataset")
                continue
            sample_trex_items[str(fact_id)] = (item['subj_label'], item['relation'], item['obj_label'])
            
        

    filtered_facts, sub2alias, obj2alias = data_filter(sample_trex_items, rel2alias, sub2alias, obj2alias, exp_mode)
    for rel in rel2alias:
        # if more than 4 alias, sample 4 alias randomly
        if len(rel2alias[rel]) > 0:
            rel2alias[rel] = random.sample(rel2alias[rel], min(4, len(rel2alias[rel])))
    for sub in sub2alias:
        if len(sub2alias[sub]) > 0:
            sub2alias[sub] = random.sample(sub2alias[sub], min(4, len(sub2alias[sub])))
    for obj in obj2alias:
        if len(obj2alias[obj]) > 0:
            obj2alias[obj] = random.sample(obj2alias[obj], min(4, len(obj2alias[obj])))
            
    print(f"😄 {len(filtered_facts.keys())} facts are retained after filtering.\n")
    all_save_dict = fun1(exp_mode, filtered_facts, 0, 1, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, max_samples)
