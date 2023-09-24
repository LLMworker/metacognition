import jsonlines
import os
import random

def main(data_path='data/chatglm_answers.jsonl',lora_path='data/small',save_path='data/eval_data.jsonl'):
    data = []
    with jsonlines.open(data_path,'r') as reader:
        for obj in reader:
            item={"lora_number":-1,"context":obj["context"],"evaluate":obj["evaluate"]}
            data.append(item)
    train_dirnames = os.listdir(f"{lora_path}/train")
    test_dirnames = os.listdir(f"{lora_path}/test")
    train_range = []
    test_range = []
    for dirname in train_dirnames:
        lora_num = int(dirname)
        train_range += list(range(100*lora_num,100*(lora_num+1)))
    for dirname in test_dirnames:
        lora_num = int(dirname)
        test_range += list(range(100*lora_num,100*(lora_num+1)))
    train_paths = [f"{lora_path}/train/{dirname}/answer.jsonl" for dirname in train_dirnames]
    test_paths = [f"{lora_path}/test/{dirname}/answer.jsonl" for dirname in test_dirnames]
    all_paths = train_paths + test_paths
    for path in all_paths:
        with jsonlines.open(path,'r') as reader:
            lora_num = int(path.split('/')[-2])
            for obj in reader:
                item={"lora_number":lora_num,"context":obj["context"],"evaluate":obj["evaluate"]}
                data.append(item)
            if 'train' in path:
                sample_range = [j for j in train_range if j not in range(100*lora_num,100*(lora_num+1))]
            else:
                sample_range = [j for j in test_range if j not in range(100*lora_num,100*(lora_num+1))]
            random_number = random.sample(sample_range, 100)
            for j in random_number:
                item={"lora_number":lora_num,"context":data[j]["context"],"evaluate":data[j]["evaluate"]}
                data.append(item)
    with jsonlines.open(save_path,'w') as writer:
        for item in data:
            writer.write(item)

if __name__=="__main__":
    main()