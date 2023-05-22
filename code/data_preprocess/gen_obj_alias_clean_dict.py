'''
Preprocess for OOV words in the object aliases.
'''
import json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast, AutoTokenizer, AutoModelForMaskedLM, TransfoXLTokenizer, TransfoXLLMHeadModel, T5Tokenizer, T5ForConditionalGeneration,OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, XLNetLMHeadModel, XLNetTokenizer, GPTNeoForCausalLM, AutoTokenizer, GPTJForCausalLM
from transformers import BloomConfig, BloomModel, OPTConfig, OPTModel, BloomTokenizerFast,GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM

from transformers import AutoTokenizer, BloomModel, BloomForCausalLM, OPTModel, OPTForCausalLM


def judge_obj_in_vocab(tokenizer, obj_label, obj_ids):

    if isinstance(tokenizer, GPT2TokenizerFast):
        reconstructed_word = "".join(
            tokenizer.convert_ids_to_tokens(obj_ids)).replace('Ġ', ' ').strip()
    elif isinstance(tokenizer, TransfoXLTokenizer):
        reconstructed_word = " ".join(
            tokenizer.convert_ids_to_tokens(obj_ids)).replace(' , ', ', ').strip()
    elif type(tokenizer).__name__ == 'GLMGPT2Tokenizer':
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, LlamaTokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, T5Tokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True).strip()
    elif isinstance(tokenizer, OpenAIGPTTokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, clean_up_tokenization_spaces=True)
    elif isinstance(tokenizer, XLNetTokenizer) or isinstance(tokenizer, GPT2Tokenizer):
        reconstructed_word = tokenizer.decode(obj_ids, skip_special_tokens=True)
    else:
        reconstructed_word = "".join(tokenizer.convert_ids_to_tokens(obj_ids)).replace('Ġ', ' ').strip()
    if isinstance(tokenizer, OpenAIGPTTokenizer) or isinstance(tokenizer, GPT2Tokenizer): 
        if (not reconstructed_word) or (reconstructed_word.lower().replace(' ','') != obj_label.lower().replace(' ','')):
            print("\tEXCLUDED object label {} not in model vocabulary\n".format(
                obj_ids
            ))
            return False
        return True
    else: 
        if (not reconstructed_word) or (reconstructed_word != obj_label):
            print("\tEXCLUDED object label {} not in model vocabulary\n".format(
                obj_ids
            ))
            return False
        return True

if __name__ == '__main__':
    device = cuda
    model_names = ['EleutherAI/gpt-neo-125M']
    for model_name in model_names:
        if 'gpt2' in model_name:
            model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
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
        elif 'bloom' in model_name:
            configuration = BloomConfig()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = BloomForCausalLM.from_pretrained(model_name).to(device)
        elif 'opt' in model_name:
            model = OPTForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(tokenizer)
        elif 'llama' in model_name:
            model = LlamaForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        elif 'glm' in model_name:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
        elif 'dolly' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        elif 'neo' in model_name:
            model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'txl' in model_name:
            tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
            model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103").to(device)
        with open("YOUR_PROJECT_PATHdata/cleaned_T_REx/allobj2alias.json",'r') as load_f:
            obj2alias = json.load(load_f)
            
        save_dict = {}
        valid_obj_num = 0
        for obj_id in obj2alias.keys():
            origial_aliases = obj2alias[obj_id]
            save_dict[obj_id] = []
            for alias in origial_aliases:
                if alias == None:
                    continue
                input_ids = tokenizer(alias, return_tensors='pt').to(device).input_ids[0]
                if judge_obj_in_vocab(tokenizer, alias, input_ids):
                    save_dict[obj_id].append(alias)
                    valid_obj_num += 1
                else: 
                    print(f"alias: {alias}, judge: {judge_obj_in_vocab(tokenizer, alias, input_ids)}")
        model_name_replaced = model_name.replace('/', '_')
        with open(f"YOUR_PROJECT_PATHdata/cleaned_T_REx/obj2alias_for_{model_name_replaced}_vocab.json", 'w') as write_f:
            json.dump(save_dict, write_f, indent=4, ensure_ascii=False)
        print(valid_obj_num / len(save_dict.keys()))