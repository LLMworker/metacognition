import chatgpt_respond
import jsonlines

def prompt_generate(question, answer, ground_truth):
    prompt = "## Question:\n" + question + "\n\n"
    prompt += "## Ground truth:\n"
    prompt += ground_truth
    prompt += "\n\n"
    prompt = prompt + "## Answer by a language model:\n" + answer +"\n\n"
    prompt += '''## Evaluate without explaination (A. clear answer B. minor error C. severe hallucination D. nonsense E. refuse to answer or other cases):\n'''
    return prompt

def evaluate(question, answer, ground_truth):
    prompt = prompt_generate(question, answer, ground_truth)
    r = 'F'
    repeat = 3
    while r == 'F' and repeat > 0:
        r = chatgpt_respond.simple_respond(prompt)
        repeat -= 1
    if len(r) == 1:
        if r in ['A','B','C','D','E']:
            result = r
        elif r in ['a','b','c','d','e']:
            result = r.upper()
        else:
            result = 'D'
    elif r[:2] == 'A.':
        result = 'A'
    elif r[:2] == 'A ':
        result = 'A'
    elif r[:2] == 'B.':
        result = 'B'
    elif r[:2] == 'B ':
        result = 'B'
    elif r[:2] == 'C.':
        result = 'C'
    elif r[:2] == 'C ':
        result = 'C'
    elif r[:2] == 'D.':
        result = 'D'
    elif r[:2] == 'D ':
        result = 'D'
    elif r[:2] == 'E.':
        result = 'E'
    elif r[:2] == 'E ':
        result = 'E'
    elif 'clear answer' in r:
        result = 'A'
    elif 'Clear answer' in r:
        result = 'A'
    elif 'Clear Answer' in r:
        result = 'A'
    elif 'minor error' in r:
        result = 'B'
    elif 'Minor error' in r:
        result = 'B'
    elif 'Minor Error' in r:
        result = 'B'
    elif 'severe hallucination' in r:
        result = 'C'
    elif 'Severe hallucination' in r:
        result = 'C'
    elif 'Severe Hallucination' in r:
        result = 'C'
    elif 'nonsense' in r:
        result = 'D'
    elif 'Nonsense' in r:
        result = 'D'
    elif 'refuse to answer' in r:
        result = 'E'
    elif 'Refuse to answer' in r:
        result = 'E'
    else:
        result = 'D'
    return result

def evaluate_dataset(ground_truth_file,answer_file):
    gtruth = {}
    evals = []
    with jsonlines.open(ground_truth_file,'r') as reader:
        for obj in reader:
            gtruth[obj['context']] = obj['target']
    with jsonlines.open(answer_file,'r') as reader:
        for obj in reader:
            evals.append(obj)
    result = {'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}
    for j,item in enumerate(evals):
        if 'evaluate' not in item.keys():
            print(f"Question {j+1}: {item['context']}\n\nAnswer:{item['answer']}\n")
            r = evaluate(item['context'],item['answer'],gtruth[item['context']])
            result[r] += 1
            evals[j]['evaluate'] = r
            if (j+1) % 100 == 0:
                with jsonlines.open(answer_file,'w') as writer:
                    for e in evals:
                        writer.write(e)
            print(f"Evaluate: {item['evaluate']}\n")
    return result

if __name__=='__main__':
    ev = evaluate_dataset('data/ground_truth_aug.jsonl','data/small/train/3/answer.jsonl')
    print(ev)