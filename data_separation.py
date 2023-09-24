import jsonlines
import random
import os

def random_train_test_split():
    train = []
    test = []
    for i in range(8):
        test += [20*i+j for j in sorted(random.sample(range(20),5))]
    train = [i for i in range(160) if i not in test]
    return train,test
    
def random_sep(ptr,sep_num,mode='section',sec_num=15):
    if mode=='section':
        secs = [ptr[i*sec_num:(i+1)*sec_num] for i in range((len(ptr))//sec_num)]
        if len(ptr) % sec_num != 0:
            secs.append(ptr[(len(ptr)//sec_num)*sec_num:])
    elif mode=='all':
        secs = [ptr]
    seps = []
    for sec in secs:
        remain = sec[:]
        while len(remain) > 0:
            new_item = sorted(random.sample(remain,min(sep_num,len(remain))))
            remain = [c for c in remain if c not in new_item]
            seps.append(new_item)
    seps.sort(key=lambda x:x[0])
    return seps

def data_sep(seps,data,output):
    all_data = []
    with jsonlines.open(data,'r') as reader:
        for item in reader:
            all_data.append(item)
    files = []
    for sep in seps:
        newdirname = '_'.join([str(n) for n in sep])
        if not os.path.exists(f'{output}/{newdirname}'):
            os.mkdir(f'{output}/{newdirname}')
        curr_data = []
        for n in sep:
            curr_data = curr_data + all_data[n*100:(n+1)*100]
        random.shuffle(curr_data)
        with jsonlines.open(f'{output}/{newdirname}/data.jsonl','w') as writer:
            for obj in curr_data:
                writer.write(obj)
        files.append(f'{output}/{newdirname}/data.jsonl')
    return files

def data_sep_all(data,output):
    train,test=random_train_test_split()
    train_seps=random_sep(train,1)
    test_seps=random_sep(test,1,mode='all')
    files = []
    if not os.path.exists(f'{output}/train'):
        os.mkdir(f'{output}/train')
    if not os.path.exists(f'{output}/test'):
        os.mkdir(f'{output}/test')
    files += data_sep(train_seps,data,f'{output}/train')
    files += data_sep(test_seps,data,f'{output}/test')
    return files


import json
from tqdm import tqdm
import argparse
import datasets
import transformers


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=True):
    model_name = "THUDM/chatglm-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/ground_truth_aug.jsonl")
    parser.add_argument("--save", type=str, default="data/small")
    parser.add_argument("--max_seq_length", type=str, default=1024)
    parser.add_argument("--skip_overlength", type=str, default=True)
    args = parser.parse_args()
    files = data_sep_all(args.data,args.save)
    for file in files:
        dataset = datasets.Dataset.from_generator(
            lambda: read_jsonl(file, args.max_seq_length, args.skip_overlength)
        )
        dataset.save_to_disk(os.path.dirname(file))

if __name__=="__main__":
    main()