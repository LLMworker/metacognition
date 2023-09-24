import os
import jsonlines
import datasets
from data_separation import read_jsonl
import subprocess

def data_organize(data_path='data/chatglm_answers.jsonl', lora_path='data/small', save_path="data/pretrain"):
    train_ptr = [int(i) for i in os.listdir(lora_path + "/train")]
    all_data = []
    with jsonlines.open(data_path,'r') as reader:
        for obj in reader:
            all_data.append(obj)
    train_data = []
    for j,item in enumerate(all_data):
        newitem = {"context":f'Evaluate<<{item["context"]}>>',"target":item["evaluate"]}
        if (j//100) in train_ptr:
            train_data.append(newitem)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with jsonlines.open(save_path + "/pretrain.jsonl",'w') as writer:
        for item in train_data:
            writer.write(item)
    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(save_path + "/pretrain.jsonl", 256)
    )
    dataset.save_to_disk(save_path)

def pretraining(path="data/pretrain"):
    args = ['python','finetune.py']
    args.append('--dataset_path')
    args.append(path)
    args.append('--lora_rank')
    args.append('16')
    args.append('--per_device_train_batch_size')
    args.append('6')
    args.append('--gradient_accumulation_steps')
    args.append('1')
    args.append('--max_steps')
    args.append('16000')
    args.append('--save_steps')
    args.append('1000')
    args.append('--save_total_limit')
    args.append('2')
    args.append('--learning_rate')
    args.append('1e-4')
    args.append('--fp16')
    args.append('--remove_unused_columns')
    args.append('false')
    args.append('--logging_steps')
    args.append('20')
    args.append('--output_dir')
    args.append(path)
    print(f'Executing command: {" ".join(args)}')
    p = subprocess.Popen(args)
    p.wait()

if __name__=="__main__":
    data_organize()
    pretraining()