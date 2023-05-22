from wikidataintegrator import wdi_core, wdi_login
import json
import random
from statistics import mean
from tqdm import tqdm
import argparse
def get_wikidata_info(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    print([item['value'] for item in all_info['aliases']['en']])
    return all_info

def get_wikidata_aliases(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    aliases = [all_info['labels']['en']['value']]
    for lang in all_info['aliases'].keys():
        if lang == 'en':
            aliases += [item['value'] for item in all_info['aliases'][lang]]
    return aliases

if __name__ == '__main__':
    print("----Loading data....")
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/sub2example_ids.json",'r') as load_f:
        sub2example_ids = json.load(load_f)
    print("----All data loaded.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash_id', type=int)
    args = parser.parse_args()
    
    sub2aliases = {}
    for sub_id in tqdm(sub2example_ids.keys()):
        if sub_id==None or len(sub_id)<1 or sub_id[0] != "Q":
            continue
        int_sub_id = int(sub_id[1:])
        if int_sub_id % 100 != args.bash_id:
            continue
        try:
            aliases = get_wikidata_aliases(sub_id)
            print(f"aliases for {sub_id}: {aliases}")
            if sub_id not in sub2aliases:
                sub2aliases[sub_id] = aliases
        except:
            print(f"error for {sub_id}")
            continue        
    with open(f"YOUR_PROJECT_PATHdata/cleaned_T_REx/sub_alias/sub2aliases_{args.bash_id}.json", 'w') as write_f:
	    json.dump(sub2aliases, write_f, indent=4, ensure_ascii=False)
     
    sub2example_ids = None
    sub2aliases = None 
     
    with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2example_ids.json",'r') as load_f:
        obj2example_ids = json.load(load_f)
    obj2aliases = {} 
    for obj_id in tqdm(obj2example_ids.keys()):
        if obj_id==None or len(obj_id)<1 or obj_id[0] != "Q":
            continue
        int_obj_id = int(obj_id[1:])
        if int_obj_id % 100 != args.bash_id:
            continue
        try:
            aliases = get_wikidata_aliases(obj_id)
            print(f"aliases for {obj_id}: {aliases}")
            if obj_id not in obj2aliases:
                obj2aliases[obj_id] = aliases
        except:
            print(f"error for {obj_id}")
            continue
    with open(f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj_alias/obj2aliases_{args.bash_id}.json", 'w') as write_f:
	    json.dump(obj2aliases, write_f, indent=4, ensure_ascii=False)