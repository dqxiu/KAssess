from dis import Instruction
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, AutoTokenizer, AutoModelForMaskedLM, TransfoXLTokenizer, TransfoXLLMHeadModel,T5Tokenizer, T5ForConditionalGeneration,OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, TransfoXLConfig, XLNetTokenizer, XLNetLMHeadModel, GPTNeoForCausalLM, AutoTokenizer, GPTJForCausalLM
from transformers import BloomConfig, BloomModel
from transformers import AutoTokenizer, BloomModel, BloomForCausalLM, OPTConfig, OPTModel, OPTForCausalLM
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
from torch.distributions.categorical import Categorical
sys.path.append('YOUR_PROJECT_PATH')
# from LAMA.lama.eval_generation_my import *
# from single_token_lama import *
import argparse
queryInstance = WikidataQueryController()
queryInstance.init_database_connection()
# parser = options.get_eval_generation_parser()
# args = options.parse_args(parser)
import nltk
import math
from nltk.corpus import stopwords
from multiprocessing.util import Finalize
from torch.multiprocessing import Pool
from torch.multiprocessing import Process, set_start_method
import multiprocessing
from multiprocessing import  Process
# from transformers import LlamaForCausalLM, LlamaTokenizer
# set_start_method('spawn')
ctx = torch.multiprocessing.get_context("spawn")
random.seed(1)


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
        try:
            parents_sub = queryInstance.get_parents(sub_id.replace('Q', ''))[0]
            parents_obj = queryInstance.get_parents(obj_id.replace('Q', ''))[0]
        except:
            continue
        
        sub_gold = sub_gold_info
        obj_gold = obj_gold_info
        
        if rel_id not in rel2alias.keys():
            print("no single text rel")
            continue
        
        rel_gold_list = rel2alias[rel_id]

        filtered_facts[fact_id] = [sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, parents_sub, parents_obj]
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
    
def condition_prob(predict_sent, given_sent, tokenizer, model, device, mode="given beta"):

    if mode == "given beta":
        # predict_sent = o
        prompt = predict_sent # s,r
        tgt_len = len(tokenizer.encode(' ' + predict_sent.replace(given_sent, '').strip()))
        encodings = tokenizer(prompt,
                              return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        return 1 / ppl.item()
    
    if mode == "given alpha":
        # predict_sent = r,o
        prompt = predict_sent # s
        tgt_len = len(tokenizer.encode(' ' + predict_sent.replace(given_sent, '').strip()))
        encodings = tokenizer(prompt,
                              return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        return 1 / ppl.item()
    

def sentence_prob(sentence,  tokenizer, model, device):

    encodings = tokenizer(sentence,
                            return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    # print(f"input_ids: {input_ids}", f"target_ids: {target_ids}")
    outputs = model(input_ids, labels=target_ids)

    ppl = torch.exp(outputs.loss)
    prob = 1 / ppl.item()
    
    return prob

def rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, parents_sub, parents_obj, max_samples):
    '''
    Given rel sub replacement
    score = 
    '''


    betas = []
    beta_temps = []
    gammas = []
    all_r_gammas = dict() # objs可以作为哪些rel的objs,同时sub也可以作为哪些rel的subs -> objs可以作为哪些sub的objs,同时rel也可以作为哪些sub的rels
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
    
    
    with torch.no_grad():
        # sum_beta [P_M(beta) * \delta(s,r) * sum_gamma P_M(gamma|beta) * \delta(o)]
        # betas represents aliases of s,r
        p_numerator_info = {}
        
        p_numerator = []
        for beta in betas:
            # P_M(beta)
            p_beta = sentence_prob(beta, tokenizer, model, device)
            p_gamma_sum = 0
            gamma_dict = dict()
            for gamma in gammas:
                # P_M(gamma|beta)
                if beta not in gamma:
                    continue
                p_gamma = condition_prob(gamma, beta, tokenizer, model, device, mode="given beta")
                gamma_dict[gamma] = round(p_gamma, 6)
                p_gamma_sum += p_gamma
            if math.isnan(p_beta) or math.isnan(p_gamma_sum):
                continue
            p_numerator.append(p_gamma_sum/len(gamma_dict.keys()))
            p_numerator_info[beta] = {'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}
        
        rr_result = {"score": p_numerator , "triplet":[sub_gold,rel_gold_list[0],obj_gold],  "p_numerator_info": p_numerator_info}
        # print(rr_result)
    return rr_result

def rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, parents_sub, parents_obj, max_samples):
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
    
    with torch.no_grad():
        p_numerator_info = dict()
        p_numerator = []
        for beta in betas:
            # P_M(beta)
            p_beta = sentence_prob(beta, tokenizer, model, device)
            p_gamma_sum = 0
            gamma_dict = dict()
            for gamma in gammas:
                # P_M(gamma|beta)
                if beta not in gamma:
                    continue
                p_gamma = condition_prob(gamma, beta, tokenizer, model, device, mode="given beta")
                gamma_dict[gamma] = round(p_gamma, 6)
                p_gamma_sum += p_gamma
            if math.isnan(p_beta) or math.isnan(p_gamma_sum):
                continue
            p_numerator.append(p_gamma_sum/len(gamma_dict.keys()))
            p_numerator_info[beta] = { 'p_beta': round(p_beta, 6), 'p_gamma_sum': round(p_gamma_sum, 6), 'gamma_dict': gamma_dict}

        rr_result = {"score": p_numerator, "triplet":[sub_gold,rel_gold_list[0],obj_gold],  "p_numerator_info": p_numerator_info}
        # print(rr_result)
    return rr_result



def fun1(exp_mode, facts, index, size, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, max_samples):
    save_dict = dict()
    for fact_id in tqdm(facts.keys()):
        fact = facts[fact_id]
        sub_id = fact[0]
        rel_id = fact[1]
        obj_id = fact[2]
        sub_gold = fact[3]
        rel_gold_list = fact[4]
        obj_gold = fact[5]
        parents_sub = fact[6]
        parents_obj = fact[7]
        if 'openai-gpt' in exp_mode:
            parents_obj = parents_obj.lower()
        if 'bs' in exp_mode:
            print("**********bs************") 
            if 'sub' in exp_mode:
                rr_bs_sub_result = rr_bs_sub_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, parents_sub, parents_obj, max_samples)
            if 'rel' in exp_mode:
                rr_bs_rel_result = rr_bs_rel_replace(tokenizer, model, device, sub_id, rel_id, obj_id, sub_gold, rel_gold_list, obj_gold, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, parents_sub, parents_obj, max_samples)
            if rr_bs_sub_result is None or rr_bs_rel_result is None:
                continue
            meta_info = {"fact_id": fact_id,"triplet": rr_bs_rel_result['triplet']}
            rr_bs_sub_result.pop('triplet')

            save_dict[str(fact_id)] = {'meta_info': meta_info, 'rr_bs_sub_result': rr_bs_sub_result, 'rr_bs_rel_result': rr_bs_rel_result}
    return save_dict



if __name__ == '__main__':
    stopwords = stopwords.words('english')
    stopwords.extend(['I', 'J', 'K', 'without'])
    device = 'cuda'
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--model_name', type=str, default="gpt2", help='which model')
    main_args = main_parser.parse_args()
    model_name = main_args.model_name
    model_name_replaced = model_name.replace('/', '_')
    # exp_mode = 'bs_sub_rel_false'
    exp_mode = 'bs_sub_rel_main' # txl or openai error stop flan-xl
    # model_name = 'openai-gpt'
    max_samples = 30 # 10, 20, 30, 40, 50
    all_save_dict = dict()
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    elif 'bloom' in model_name:
        configuration = BloomConfig()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BloomForCausalLM.from_pretrained(model_name).to(device)
    elif 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name).to(device)
    elif 'llama' in model_name:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    elif 'bert' in model_name:
        model = BertForMaskedLM.from_pretrained(model_name).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif 't5' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif 'openai-gpt' in model_name:
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    elif 'xlnet' in model_name:
        model = XLNetLMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
    elif 'gpt-j' in model_name:
        model = GPTJForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif 'neo' in model_name:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103").to(device)
    model.eval()

    stopwords_ids = [tokenizer.encode(' '+ stopword)[0] for stopword in stopwords]
    
    print("⏰ Loading data....")
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",'r') as load_f:
        rel2example_ids = json.load(load_f)
    
    if 'false' not in exp_mode and 'para' not in exp_mode and 'orig' not in exp_mode and 'freq' not in exp_mode and 'main' not in exp_mode and 'human' not in exp_mode:
        with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/example_1000test.json",'r') as load_f:
            all_trex = json.load(load_f)
    else:
        if 'freq' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx_freq"
        elif 'false_p' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx_false_p"
        elif 'para' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx_para"
        elif 'orig' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx"
        elif 'false' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx_false"
        elif 'main' in exp_mode:
            rootdir = "YOUR_PROJECT_PATHLAMA/data/my_TREx_main_new_test"
            
        all_trex = []
        list_path = os.listdir(rootdir)
        for i in range(0, len(list_path)):
            # 构造路径
            path = os.path.join(rootdir, list_path[i])
            with jsonlines.open(path) as reader:
                for obj in reader:
                    all_trex.append(obj)
    
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
    if 'gpt2' in model_name_replaced:
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_gpt2_vocab.json"
    elif 't5' in model_name_replaced:
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_t5-large_vocab.json"
    elif 'bloom' in model_name_replaced:
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_bigscience_bloom-560m_vocab.json"
    elif 'opt' in model_name_replaced:
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_facebook_opt-125m_vocab.json"
    elif 'llama' in model_name_replaced:
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_decapoda-research_llama-7b-hf_vocab.json"
    else: 
        vocab_path = f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_{model_name_replaced}_vocab.json"

    with open(vocab_path,'r') as load_f:
        obj2alias = json.load(load_f)
        if 'openai-opt' in model_name:
            for obj_id in obj2alias.keys():
                cur_obj_aliases = obj2alias[obj_id]
                for obj_alias_id in range(len(cur_obj_aliases)):
                    cur_obj_aliases[obj_alias_id] = cur_obj_aliases[obj_alias_id].lower()
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allsub2alias.json",'r') as load_f:
        sub2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2rel2rate.json",'r') as load_f:
        obj2rel2rate = json.load(load_f)
    print("😄 All data loaded.\n")
    

    # rel alias filter
    with open("YOUR_PROJECT_PATHrelation2template.json",
                'r') as load_f:
        rel2alias = json.load(load_f)
    

        
                    
    para_alias = dict()
    if 'para' in exp_mode:
        para_name = exp_mode[-1]
        if para_name == 'a':
            para_name = ''
        if para_name != '':
            with jsonlines.open(f'YOUR_PROJECT_PATHLAMA/data/relations{para_name}.jsonl', 'r') as reader:
                for line in reader:
                    line_rel =  line["relation"]
                    rel2alias[line_rel][random.sample(rel2alias[line_rel].keys(),1)[0]] = line["template"]
                
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
    # fun1(exp_mode, filtered_facts, 0, 1, tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, max_samples)
    process_list = []
    num_process = 4
    p = ctx.Pool(num_process)
    
    for i in range(num_process):
        cur_facts = dict()
        for fact_id in filtered_facts.keys():
            if int(fact_id) % num_process == i:
                cur_facts[fact_id] = filtered_facts[fact_id]

        process_list.append(p.apply_async(fun1, args=(exp_mode, cur_facts, i, num_process,tokenizer, model, device, sub2alias, rel2alias, obj2alias, obj2rel2rate, rel2sub2rate, max_samples)))
        # p = Process(target=fun1,args=(facts)) 
        # p.start()
        # process_list.append(p)
        
    p.close()
    p.join()
    for i in process_list:
        i_save_dict = i.get()
        for fact_id in i_save_dict.keys():
            all_save_dict[fact_id] = i_save_dict[fact_id]

    print('done')
    with open(f"YOUR_PROJECT_PATHscores_kprompts/{exp_mode}_{model_name_replaced}_{max_samples}_orig.json", 'w') as write_f:
        json.dump(all_save_dict, write_f, indent=4, ensure_ascii=False)