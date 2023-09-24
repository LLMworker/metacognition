import jsonlines
import os
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

lora_path = "data/small/test"
lora_ids = os.listdir(lora_path)
eval_path = "data/eval"
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

all_data = []
with jsonlines.open("data/eval_data.jsonl",'r') as reader:
    for obj in reader:
        all_data.append(obj)

ref_data={}
eval_data={}
for item in all_data:
    if item["lora_number"]==-1:
        ref_data[item["context"]] = item["evaluate"]
    else:
        if str(item["lora_number"]) not in eval_data.keys():
            eval_data[str(item["lora_number"])] = [item]
        else:
            eval_data[str(item["lora_number"])].append(item)
        eval_data[str(item["lora_number"])][-1]["ref"] = ref_data[item["context"]]

def test_without_lora():
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model = PeftModel.from_pretrained(model, eval_path)
    output = {'A':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'B':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'C':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'D':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'E':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}}
    for lora_id in lora_ids:
        print(f"Working on LoRA {lora_id}...")
        curr_data = eval_data[lora_id]
        for j,item in enumerate(curr_data):
            print(f"Question {j+1}: {item['context']}")
            response,history=model.chat(tokenizer,item["context"],history=[])
            print(f"Evaluation: {response}")
            if response in ['A','B','C','D','E']:
                output[item['ref']][response] += 1
            else:
                output[item['ref']]['F'] += 1
        print(output)
    return output

def test_with_lora():
    output = {'A':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'B':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'C':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'D':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0},
              'E':{'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}}
    for lora_id in lora_ids:
        print(f"Working on LoRA {lora_id}...")
        model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        model = PeftModel.from_pretrained(model, f"{lora_path}/{lora_id}")
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, eval_path)
        curr_data = eval_data[lora_id]
        for j,item in enumerate(curr_data):
            response,history=model.chat(tokenizer,item["context"],history=[])
            print(f"Question {j+1}: {item['context']}")
            response,history=model.chat(tokenizer,item["context"],history=[])
            print(f"Evaluation: {response}")
            if response in ['A','B','C','D','E']:
                output[item['evaluate']][response] += 1
            else:
                output[item['evaluate']]['F'] += 1
        print(output)
        del model
    return output

def confmatrix(output):
    result = {'T':{'T':0,'F':0},'F':{'T':0,'F':0}}
    for key in output.keys():
        if key in ['C','D']:
            k1 = 'T'
        else:
            k1 = 'F'
        for k in output[key].keys():
            if k in ['C','D']:
                k2 = 'T'
            else:
                k2 = 'F'
            result[k1][k2] += output[key][k]
    return result

def average_distance(output):
    cor = {'A':5,'B':4,'C':2,'D':1,'E':3, 'F':0}
    N = 0
    D = 0
    for key in output.keys():
        s1 = cor[key]
        for k in output[key].keys():
            s2 = cor[k]
            d12 = abs(s1-s2)
            num = output[key][k]
            N += num
            D += (num * d12)
    return D/N

if __name__=="__main__":
    result_1 = test_without_lora()
    # result_2 = test_with_lora()
    print(result_1)
    # print(result_2)
    print(confmatrix(result_1))
    print(average_distance(result_1))
    # print(confmatrix(result_2))
    # print(average_distance(result_2))
