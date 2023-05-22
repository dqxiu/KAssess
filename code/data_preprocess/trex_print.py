from dis import Instruction
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from databaseconnection import *
from wikidata_get import *
# from wikidataintegrator import wdi_core, wdi_login
import random
from statistics import mean
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


# set seed
random.seed(1)


queryInstance = WikidataQueryController()
queryInstance.init_database_connection()


def super_class_cmp_score(sub_id, rel_text, obj_id, tokenizer, model, device):
    '''
    Given rel, sub replacement
    score = gold fact prob - superclass sub fact prob
    '''
    # rel_text = 'was born in'
    # rel_text = get_wikidata_info(rel_id)['labels']['en']['value']
    prompt = '{} ' + rel_text + ' {}'

    try:
        sub_gold = queryInstance.get_entity(sub_id.replace('Q', ''))[1]
        parents_sub = queryInstance.get_parents(sub_id.replace('Q', ''))[0]
        obj_gold = queryInstance.get_entity(obj_id.replace('Q', ''))[1]
        parents_obj = queryInstance.get_parents(obj_id.replace('Q', ''))[0]
    except:
        return None

    with torch.no_grad():
        tgt_len = len(tokenizer.encode(' ' + obj_gold))
        print(f"----{sub_gold}, {rel_text}, {obj_gold}----")
        # the fact
        encodings = tokenizer(prompt.format(sub_gold, obj_gold),
                              return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        print(
            f'Original fact prob    {sub_gold} {rel_text} {obj_gold}: {(1 /ppl.item()):.6f}'
        )

        # the parent fact
        encodings = tokenizer("A " + prompt.format(parents_sub[1], obj_gold),
                              return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        calibrate = torch.exp(outputs.loss)
        print(
            f'Parent fact prob    A/an {parents_sub[1]} {rel_text} {obj_gold}: {(1 / calibrate.item()):.6f}'
        )

        print(
            f'Superclass_score:  {(1 / ppl.item() - 1 / calibrate.item()):.6f}'
        )
        print()


def neighbor_class_cmp_score(sub_id,
                             rel_text,
                             obj_id,
                             tokenizer,
                             model,
                             device,
                             neighbor_mode='same parent objs',
                             relsubs=None):
    '''
    Given rel, sub replacement
    score = gold fact prob - avg(neighbor/same type sub fact probs) 
    todo: expectation; risk: the parent2subject relation is same to the relation
    '''
    # rel_text = 'was born in'
    # rel_text = get_wikidata_info(rel_id)['labels']['en']['value']

    prompt = '{} ' + rel_text + ' {}'

    try:
        sub_gold = queryInstance.get_entity(sub_id.replace('Q', ''))[1]
        parents_sub = queryInstance.get_parents(sub_id.replace('Q', ''))[0]
        obj_gold = queryInstance.get_entity(obj_id.replace('Q', ''))[1]
        parents_obj = queryInstance.get_parents(obj_id.replace('Q', ''))[0]
    except:
        return None

    if neighbor_mode == 'same parent objs':
        sub_candidates_all = queryInstance.get_children(parents_sub[0])
    elif neighbor_mode == 'all possible subs':
        sub_candidates_all = relsubs

        #sample relsubs
    min_sub = min(len(sub_candidates_all), 1000)
    sub_candidates = random.sample(sub_candidates_all, min_sub)

    with torch.no_grad():

        tgt_len = len(tokenizer.encode(' ' + obj_gold))
        print(f"----{sub_gold}, {rel_text}, {obj_gold}----")
        # the fact
        encodings = tokenizer(prompt.format(sub_gold, obj_gold),
                              return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-tgt_len] = -100
        outputs = model(input_ids, labels=target_ids)
        ppl = torch.exp(outputs.loss)
        print(
            f'Original fact prob    {sub_gold} {rel_text} {obj_gold}: {(1 /ppl.item()):.6f}'
        )

        # sub_candidates = [sub[1] for sub in sub_candidates if sub[1] != sub_gold]
        sub_candidates = [sub[1] for sub in sub_candidates]
        calibrate_list = []
        for sub_candi in sub_candidates:
            # the neighbor facts
            # print(sub_candi)
            encodings = tokenizer("A " + prompt.format(sub_candi, obj_gold),
                                  return_tensors='pt')
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-tgt_len] = -100
            outputs = model(input_ids, labels=target_ids)
            calibrate = torch.exp(outputs.loss)
            calibrate_list.append(1 / calibrate.item())

        print(
            f'Neighbor fact avg     {sub_candi} (just one case) {rel_text} {obj_gold}: {mean(calibrate_list):.6f}'
        )

        print(
            f'Neighbor fact num:  {len(sub_candidates_all)},  Neighbor fact prob sum:  {sum(calibrate_list) * len(sub_candidates_all) / min_sub :.6f}'
        )

        print(
            f'Neighbor_score:  {(1 / ppl.item() - mean(calibrate_list)):.6f}')
        print()


def mutual_info(sub_id,
                rel_text,
                obj_id,
                tokenizer,
                model,
                device,
                neighbor_mode='all possible subs',
                relsubs=None):
    '''
    Given rel, sub replacement
    score = sum P(sub | rel) * P(obj | sub, rel)
    '''

    prompt = '{} ' + rel_text + ' {}'

    sub_gold = queryInstance.get_entity(sub_id.replace('Q', ''))[1]
    parents_sub = queryInstance.get_parents(sub_id.replace('Q', ''))[0]
    obj_gold = queryInstance.get_entity(obj_id.replace('Q', ''))[1]
    parents_obj = queryInstance.get_parents(obj_id.replace('Q', ''))[0]

    if neighbor_mode == 'same parent objs':
        sub_candidates = queryInstance.get_children(parents_sub[0])
    elif neighbor_mode == 'all possible subs':
        sub_candidates = relsubs

def count_save_entities(sub2example_ids, obj2example_ids):
    save_subs = dict()
    save_objs = dict()
    for sub in sub2example_ids.keys():
        if sub not in save_subs.keys():
            try:
                gold_sub = queryInstance.get_entity(sub.replace('Q', ''))[1]
                save_subs[sub] = gold_sub
            except:
                continue
    for obj in obj2example_ids.keys():
        if obj not in save_objs.keys():
            try:
                gold_obj = queryInstance.get_entity(obj.replace('Q', ''))[1]
                save_objs[obj] = gold_obj
            except:
                continue

    with open("data/cleaned_T_REx/subs2singletext.json", "w") as fout:
        json.dump(save_subs, fout, indent=4, ensure_ascii=False)
    print(f"num of subs: {len(save_subs.keys())}")
    with open("data/cleaned_T_REx/objs2singletext.json", "w") as fout:
        json.dump(save_objs, fout, indent=4, ensure_ascii=False)
    print(f"num of objs: {len(save_objs.keys())}")
    return None

queryInstance = WikidataQueryController()
queryInstance.init_database_connection()
def count_save_rel2sub2rate(sub2example_ids, relation2example_ids, all_trex):
    save_dict = dict()
    simple_ratio = dict()
    
    for item_dict in tqdm(all_trex):
        subj_label = item_dict['subj_label']
        relation = item_dict['relation']
        if relation not in save_dict.keys():
            save_dict[relation] = dict()
            
        if subj_label not in save_dict[relation].keys():
            save_dict[relation][subj_label] = 1
        else:
            save_dict[relation][subj_label] += 1
        if "cur_rel_total" not in save_dict[relation].keys():
            save_dict[relation]["cur_rel_total"] = 1
        else:
            save_dict[relation]["cur_rel_total"] += 1
    print("finish counting")
    
    for rel in tqdm(save_dict.keys()):
        if rel not in simple_ratio.keys():
            simple_ratio[rel] = dict()
        
        rel_num = save_dict[rel]["cur_rel_total"]
        if rel_num == 0:
            continue
        for subj in save_dict[rel].keys():
            if subj == "cur_rel_total":
                continue
            if subj not in simple_ratio[rel].keys():
                simple_ratio[rel][subj] = save_dict[rel][subj] / rel_num
                
        # if the current rel has less than 4 subs in simple_ratio, randomly sample the rest from the whole subs list, and the simple_ratio of each sampled ones are (1-current_ratio_sum)/whole_subs_num
        # if len(simple_ratio[rel].keys()) < 4:
        #     cur_ratio_sum = sum(simple_ratio[rel].values())
        #     rest_num = 4 - len(simple_ratio[rel].keys())
        #     for subj in random.sample(list(sub2example_ids.keys()), rest_num):
        #         simple_ratio[rel][subj] = 1-cur_ratio_sum) / len(sub2example_ids.keys())

    with open("data/cleaned_T_REx/rel2sub2rate.json", "w") as fout:
        json.dump(simple_ratio, fout, indent=4, ensure_ascii=False)

    return None



def count_save_sub2rel2rate(sub2example_ids, relation2example_ids, all_trex):
    save_dict = dict()
    simple_ratio = dict()

    for item_dict in tqdm(all_trex):
        subj_label = item_dict['subj_label']
        relation = item_dict['relation']
        if subj_label not in save_dict.keys():
            save_dict[subj_label] = dict()

        if relation not in save_dict[subj_label].keys():
            save_dict[subj_label][relation] = 1
        else:
            save_dict[subj_label][relation] += 1

        if "cur_sub_total" not in save_dict[subj_label].keys():
            save_dict[subj_label]["cur_sub_total"] = 1
        else:
            save_dict[subj_label]["cur_sub_total"] += 1

    print("finish counting")

    for subj in tqdm(save_dict.keys()):
        if subj not in simple_ratio.keys():
            simple_ratio[subj] = dict()

        sub_num = save_dict[subj]["cur_sub_total"]
        if sub_num == 0:
            continue

        for rel in save_dict[subj].keys():
            if rel == "cur_sub_total":
                continue

            if rel not in simple_ratio[subj].keys():
                simple_ratio[subj][rel] = save_dict[subj][rel] / sub_num

        # if the current subj has less than 4 rels in simple_ratio, randomly sample the rest from the whole rels list, and the simple_ratio of each sampled ones are (1-current_ratio_sum)/whole_rels_num
        # if len(simple_ratio[subj].keys()) < 4:
        #     cur_ratio_sum = sum(simple_ratio[subj].values())
        #     rest_num = 4 - len(simple_ratio[subj].keys())
        #     for rel in random.sample(list(relation2example_ids.keys()), rest_num):
        #         simple_ratio[subj][rel] = (1-cur_ratio_sum) / len(relation2example_ids.keys())

    with open("data/cleaned_T_REx/sub2rel2rate.json", "w") as fout:
        json.dump(simple_ratio, fout, indent=4, ensure_ascii=False)

    return None



def obj2rel(obj2example_ids, relation2example_ids, all_trex):
    save_dict = dict() 
    
    for item_dict in tqdm(all_trex):
        # if len(save_dict.keys()) > 50:
        #     break
        obj_label = item_dict['obj_label']
        relation = item_dict['relation']
        if obj_label not in save_dict.keys():
            save_dict[obj_label] = []
        if relation not in save_dict[obj_label]:
            save_dict[obj_label].append(relation)
    
    with open("data/cleaned_T_REx/obj2rel.json", "w") as fout:
        json.dump(save_dict, fout, indent=4, ensure_ascii=False)
    return None

def count_save_obj2sub2rate(obj2example_ids, sub2example_ids, all_trex, invalid_sub):
    save_dict = dict()
    simple_ratio = dict()
    for item_dict in tqdm(all_trex):
        subj_label = item_dict['subj_label']
        obj_label = item_dict['obj_label']
        if obj_label not in save_dict.keys():
            save_dict[obj_label] = dict()
        if subj_label in sub2example_ids.keys() and subj_label not in invalid_sub:
            if subj_label not in save_dict[obj_label].keys():
                save_dict[obj_label][subj_label] = 1
            else:
                save_dict[obj_label][subj_label] += 1
            if "cur_obj_total" not in save_dict[obj_label].keys():
                save_dict[obj_label]["cur_obj_total"] = 1
            else:
                save_dict[obj_label]["cur_obj_total"] += 1

    num_enough = 0
    for obj_label in tqdm(save_dict.keys()):
        if len(save_dict[obj_label].keys()) >= 5:
            num_enough += 1
            continue
        try: 
            parents_obj = queryInstance.get_parents(obj_label.replace('Q', ''))[0]
            obj_candidates_all = queryInstance.get_children(parents_obj[0])
            obj_candidates_all = [obj[0] for obj in obj_candidates_all if obj[0] != obj_label]
            obj_neighbor_subjs = []
            for obj_candidate in obj_candidates_all:
                obj_candidate_label = "Q" + str(obj_candidate)
                if obj_candidate_label in save_dict.keys():
                    for subj in save_dict[obj_candidate_label].keys():
                        if subj != "cur_obj_total" and subj not in save_dict[obj_label].keys():
                            obj_neighbor_subjs.append(subj)
            obj_neighbor_subjs = list(set(obj_neighbor_subjs))
        except:
            obj_neighbor_subjs = []
        if len(obj_neighbor_subjs) > 0:
            # sample the rest from obj_candidates_all to save_dict[obj_label]
            sampled_subjs = random.sample(obj_neighbor_subjs, min(5 - len(save_dict[obj_label].keys()), len(obj_neighbor_subjs)))
            for subj in sampled_subjs:
                if subj in sub2example_ids.keys() and subj not in invalid_sub:
                    save_dict[obj_label][subj] = 0.5
                    if "cur_obj_total" not in save_dict[obj_label].keys():
                        save_dict[obj_label]["cur_obj_total"] = 0.5
                    else:
                        save_dict[obj_label]["cur_obj_total"] += 0.5
        
        # if still less than 4, sample the rest from the whole subjs list randomly
        if len(save_dict[obj_label].keys()) < 5:
            cur_subj_num = len(save_dict[obj_label].keys())
            rest_num = 5 - cur_subj_num
            unused_subjs = list(set(sub2example_ids.keys()) - set(save_dict[obj_label].keys()))
            for subj in random.sample(unused_subjs, rest_num):
                if subj in sub2example_ids.keys() and subj not in invalid_sub:
                    save_dict[obj_label][subj] = 0.25
                    if "cur_obj_total" not in save_dict[obj_label].keys():
                        save_dict[obj_label]["cur_obj_total"] = 0.25
                    else:
                        save_dict[obj_label]["cur_obj_total"] += 0.25

    print("enough ratio: ", num_enough / len(save_dict.keys()))

    print("finish counting")
    for obj in tqdm(save_dict.keys()):
        if obj not in simple_ratio.keys():
            simple_ratio[obj] = dict()
        obj_num = save_dict[obj]["cur_obj_total"]
        if obj_num == 0:
            continue
        for subj in save_dict[obj].keys():
            if subj == "cur_obj_total":
                continue
            if subj not in simple_ratio[obj].keys():
                simple_ratio[obj][subj] = save_dict[obj][subj] / obj_num
    with open("data/cleaned_T_REx/obj2sub2rate.json", "w") as fout:
        json.dump(simple_ratio, fout, indent=4, ensure_ascii=False)
    return None


def count_save_obj2rel2rate(obj2example_ids, relation2example_ids, all_trex, rel2alias):
    save_dict = dict()
    simple_ratio = dict()
    for item_dict in tqdm(all_trex):
        obj_label = item_dict['obj_label']
        relation = item_dict['relation']
        if obj_label not in save_dict.keys():
            save_dict[obj_label] = dict()
        if relation in rel2alias.keys():
            if relation not in save_dict[obj_label].keys():
                save_dict[obj_label][relation] = 1
            else:
                save_dict[obj_label][relation] += 1
            if "cur_obj_total" not in save_dict[obj_label].keys():
                save_dict[obj_label]["cur_obj_total"] = 1
            else:
                save_dict[obj_label]["cur_obj_total"] += 1
            
    num_enough = 0
    for obj_label in tqdm(save_dict.keys()):
        if len(save_dict[obj_label].keys()) >= 5:
            num_enough += 1
            continue
        try: 
            parents_obj = queryInstance.get_parents(obj_label.replace('Q', ''))[0]
            obj_candidates_all = queryInstance.get_children(parents_obj[0])
            obj_candidates_all = [obj[0] for obj in obj_candidates_all if obj[0] != obj_label]
            obj_neighbor_rels = []
            for obj_candidate in obj_candidates_all:
                obj_candidate_label = "Q" + str(obj_candidate)
                if obj_candidate_label in save_dict.keys():
                    for rel in save_dict[obj_candidate_label].keys():
                        if rel != "cur_obj_total" and rel not in save_dict[obj_label].keys():
                            obj_neighbor_rels.append(rel)
            obj_neighbor_rels = list(set(obj_neighbor_rels))
        except:
            obj_neighbor_rels = []
        if len(obj_neighbor_rels) > 0:
            sampled_rels = random.sample(obj_neighbor_rels, min(5 - len(save_dict[obj_label].keys()), len(obj_neighbor_rels)))
            for rel in sampled_rels:
                if rel in rel2alias.keys():
                    save_dict[obj_label][rel] = 0.5
                    if "cur_obj_total" not in save_dict[obj_label].keys():
                        save_dict[obj_label]["cur_obj_total"] = 0.5
                    else:
                        save_dict[obj_label]["cur_obj_total"] += 0.5
        if len(save_dict[obj_label].keys()) < 5:
            cur_rel_num = len(save_dict[obj_label].keys())
            rest_num = 5 - cur_rel_num
            used_rels = list(set(relation2example_ids.keys()) - set(save_dict[obj_label].keys()))
            for rel in random.sample(used_rels, rest_num):
                if rel in rel2alias.keys():
                    save_dict[obj_label][rel] = 0.25
                    if "cur_obj_total" not in save_dict[obj_label].keys():
                        save_dict[obj_label]["cur_obj_total"] = 0.25
                    else:
                        save_dict[obj_label]["cur_obj_total"] += 0.25

    print("enough ratio: ", num_enough / len(save_dict.keys()))            

    print("finish counting")
    for obj in tqdm(save_dict.keys()):
        if obj not in simple_ratio.keys():
            simple_ratio[obj] = dict()
        obj_num = save_dict[obj]["cur_obj_total"]
        if obj_num == 0:
            continue
        for rel in save_dict[obj].keys():
            if rel == "cur_obj_total":
                continue
            if rel not in simple_ratio[obj].keys():
                simple_ratio[obj][rel] = save_dict[obj][rel] / obj_num
    with open("data/cleaned_T_REx/obj2rel2rate.json", "w") as fout:
        json.dump(simple_ratio, fout, indent=4, ensure_ascii=False)
    return None


if __name__ == '__main__':
    device = 'cuda'
    model_name = 'gpt2-xl'

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    print("⏰ Loading data....")
    with open(
            "YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",
            'r') as load_f:
        sub2example_ids = json.load(load_f)
    with open(
            "YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",
            'r') as load_f:
        obj2example_ids = json.load(load_f)
    with open(
            "YOUR_PROJECT_PATHdata/cleaned_T_REx/relation2example_ids.json",
            'r') as load_f:
        relation2example_ids_raw = json.load(load_f)
    with open(
            "YOUR_PROJECT_PATHdata/cleaned_T_REx/example.json",
            'r') as load_f:
        all_trex = json.load(load_f)
    with open("YOUR_PROJECT_PATHcode/rel_dict.json", 'r') as load_f:
        rel_dict = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/rel2sub_ids.json",
              'r') as load_f:
        rel2sub_ids = json.load(load_f)
    with open("YOUR_PROJECT_PATHrelation2template.json",
                'r') as load_f:
        rel2alias = json.load(load_f)
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allsub2alias.json",'r') as load_f:
        sub2alias = json.load(load_f)
    
    invalid_sub = []
    for sub in sub2example_ids.keys():
        if sub not in sub2alias.keys():
            invalid_sub.append(sub)
        elif len(sub2alias[sub]) == 0:
            invalid_sub.append(sub)
    print("😄 All data loaded.\n")
    
    relation2example_ids = dict()
    for rel_id in relation2example_ids_raw.keys():
        if rel_id in rel2alias.keys():
            relation2example_ids[rel_id] = relation2example_ids_raw[rel_id]
    count_save_obj2rel2rate(obj2example_ids, relation2example_ids, all_trex, rel2alias)
    count_save_obj2sub2rate(obj2example_ids, sub2example_ids, all_trex, invalid_sub)
