import json
import jsonlines
import os
import numpy as np

def data_statistics():
    print("⏰ Loading data....")
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",'r') as load_f:
        rel2example_ids = json.load(load_f)
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
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allobj2alias.json",'r') as load_f:
        obj2alias = json.load(load_f)
    print("😄 All data loaded.\n")


    # rel alias filter
    rel2alias = dict()
    for relid in rel_dict.keys():
        if rel_dict[relid]!= None:
            for rel_alias in rel_dict[relid].keys():
                if rel_dict[relid][rel_alias] != "" and rel_dict[relid][rel_alias][-4:] == "[Y]." and rel_dict[relid][rel_alias][0:4] == "[X] ":
                    if relid not in rel2alias.keys() and len(rel_alias)>0:
                        rel2alias[relid] = dict()
                    if len(rel_alias)>0:
                        rel2alias[relid][rel_alias] = rel_dict[relid][rel_alias]
                        
    entity2example_ids = dict()
    for subid in sub2example_ids.keys():
        if subid not in entity2example_ids.keys():
            entity2example_ids[subid] = sub2example_ids[subid]
    for objid in obj2example_ids.keys():
        if objid not in entity2example_ids.keys():
            entity2example_ids[objid] = obj2example_ids[objid]
    # entities 
    entity2alias = dict()
    for subid in sub2alias.keys():
        if subid not in entity2alias.keys():
            entity2alias[subid] = sub2alias[subid]
    for objid in obj2alias.keys():
        if objid not in entity2alias.keys():
            entity2alias[objid] = obj2alias[objid]
         
    print("The following are the statistical information of the datasets")
    print("The number of all relations: ", len(rel2example_ids))
    print("The number of all subjects: ", len(sub2example_ids))
    print("The number of all objects: ", len(obj2example_ids))
    print("The number of all entities: ", len(entity2example_ids))
    print("The number of all relations with alias: ", len(rel2alias))
    print("The number of all subjects with alias: ", len(sub2alias))
    print("The number of all objects with alias: ", len(obj2alias))
    print("The number of all objects with relation: ", len(obj2rel))
    print("The number of relation aliaes: ", sum([len(rel2alias[relid]) for relid in rel2alias.keys()]))
    print("The number of subject aliaes: ", sum([len(sub2alias[subid]) for subid in sub2alias.keys()]))
    print("The number of object aliaes: ", sum([len(obj2alias[objid]) for objid in obj2alias.keys()]))
    print("The number of all entities aliaes: ", sum([len(entity2alias[entityid]) for entityid in entity2alias.keys()]))

def model_statistic():

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, AutoTokenizer, AutoModelForMaskedLM, TransfoXLTokenizer, TransfoXLLMHeadModel,T5Tokenizer, T5ForConditionalGeneration,OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, TransfoXLConfig, XLNetTokenizer, XLNetLMHeadModel, GPTNeoForCausalLM, AutoTokenizer, GPTJForCausalLM, BloomModel, BloomForCausalLM, OPTConfig, OPTModel, OPTForCausalLM


    def count_params(model_name,model):
        # param_sum = 0
        with open('models.txt', 'w') as fm:
            fm.write(str(model))
        # 计算模型的参数总量
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    
        print('{} has {} M parameters'.format(model_name, params / 1e6))
        
    for model_name in ['bigscience/bloom-1b1','facebook/opt-1.3b','','t5-base','t5-large', 'EleutherAI/gpt-neo-125M','EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B','gpt2', 'gpt2-xl', 'xlnet-base-cased', 'xlnet-large-cased','txl','openai-gpt']:
        if 'gpt2' in model_name:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        elif 'bert' in model_name:
            model = BertForMaskedLM.from_pretrained(model_name)
        elif 't5' in model_name:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'openai-gpt' in model_name:
            model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
        elif 'xlnet' in model_name:
            model = XLNetLMHeadModel.from_pretrained(model_name)
        elif 'gpt-j' in model_name:
            model = GPTJForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'neo' in model_name:
            model = GPTNeoForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'opt' in model_name:
            model = OPTForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'bloom' in model_name:
            model = BloomForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        count_params(model_name,model)

def quiz_statistics():
    mode = 'bs'
    with open('YOUR_PROJECT_PATHscores/multiprocess_main_bs_sub_rel_main_gpt2_30_all_info.json', 'r') as load_f:
        load_dict = json.load(load_f)
    if 'simple' in mode:
        sub_replace_result_name = 'rr_sub_result'
        rel_replace_result_name = 'rr_rel_result'
    elif 'bs' in mode:
        sub_replace_result_name = 'rr_bs_sub_result'
        rel_replace_result_name = 'rr_bs_rel_result'

    sub_nu_sents = 0
    sub_de_sents = 0
    valid_num = 0
    for fact_id in load_dict:
        sub_replace = load_dict[fact_id][sub_replace_result_name]
        sub_nu_prompt_dict = sub_replace['p_numerator_info']
        sub_nu_sents += len(sub_nu_prompt_dict['gammas'])
        sub_de_prompt_dict = sub_replace['p_denominator_info']
        sub_de_sents += len(sub_de_prompt_dict['gammas'])
        valid_num += 1

    print(f"average sub nu sent num: {sub_nu_sents/valid_num}"), print(f"average sub de sent num: {sub_de_sents/valid_num}")
if __name__ == '__main__':
    # model_statistic()
    # find_case()
    data_statistics()
    # quiz_statistics()