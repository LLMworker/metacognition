from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

def answer(question, lora = None):
    if lora:
        mymodel = PeftModel.from_pretrained(model,lora)
    else:
        mymodel = model
    response, history = mymodel.chat(tokenizer, question, history=[])
    return response


if __name__=='__main__':
    ans = answer("Question about Astrophysics: What is the Hubble Law?")
    print(ans)
    print('\n\n')