import jsonlines
import argparse
import os

def gather(data_file, save_file, model = 'chatglm6b', lora=None):
    if model == 'chatglm6b':
        from chatglm6b_answer import answer
    data = []
    if not os.path.exists(save_file):
        with jsonlines.open(data_file,'r') as reader:
            for obj in reader:
                data.append(obj)
    else:
        with jsonlines.open(save_file,'r') as reader:
            for obj in reader:
                data.append(obj)        
    for i,item in enumerate(data):
        if "answer" not in item.keys():
            q = item["context"]
            print(f"Question {i+1}: {q}")
            a = answer(q, lora)
            if lora:
                print(f"Answer with lora {lora}: {a}")
            else:
                print(f"Answer: {a}")
            myanswer = a
            data[i]['answer'] = myanswer
            with jsonlines.open(save_file,'w') as writer:
                for obj in data:
                    writer.write(obj)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Gather answers for questions.")
    parser.add_argument('--data',type=str,help='Data file path.')
    parser.add_argument('--save',type=str,help='Save path')
    parser.add_argument('--model',type=str,default='chatglm6b',help='Model name')
    parser.add_argument('--lora',type=str,default=None,help='LoRA path')
    args = parser.parse_args()
    gather(args.data,args.save,args.model,args.lora)