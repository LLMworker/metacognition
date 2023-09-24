import os
import subprocess

def gather_all(datapath='small'):
    train_path = os.listdir(f'{datapath}/train')
    test_path = os.listdir(f'{datapath}/test')
    for j,path in enumerate(train_path + test_path):
        print(f"Gathering answers of {j+1} th LoRA: {path}...")
        if path in train_path:
            train_or_test = 'train'
        else:
            train_or_test = 'test'
        args = ['python','answer_gathering.py']
        args.append('--data')
        args.append(f"{datapath}/{train_or_test}/{path}/data.jsonl")
        args.append('--save')
        args.append(f"{datapath}/{train_or_test}/{path}/answer.jsonl")
        args.append('--lora')
        args.append(f"{datapath}/{train_or_test}/{path}")
        p = subprocess.Popen(args)
        p.wait()

if __name__=="__main__":
    gather_all()