import random
import jsonlines
import argparse

aug_prompts = ["","Please answer a question about \{topic\}: ","Here is a question about \{topic\}: ","Question about \{topic\}: "]

def DataAugmentation(items):
    result = []
    for item in items:
        t = item['topic']
        q = item['question']
        g = item['ground_truth']
        aug = random.choice(aug_prompts).replace("\{topic\}",t)
        result += [{"topic":t,"context":aug + q,"target":g}]
    return result

def aug_data(data_file,save_file):
    data = []
    with jsonlines.open(data_file,'r') as reader:
        for obj in reader:
            data.append(obj)
    result = DataAugmentation(data)
    with jsonlines.open(save_file,'w') as writer:
        for obj in result:
            writer.write(obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple ata augmentation by adding random prefix to questions.')
    parser.add_argument('--data',type=str,default='data/ground_truth.jsonl',help='The path of the original data file.')
    parser.add_argument('--save',type=str,default='data/ground_truth_aug.jsonl',help='The path to save converted files.')
    args = parser.parse_args()
    aug_data(args.data,args.save)