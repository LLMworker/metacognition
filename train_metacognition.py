import jsonlines
import os
import shutil
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import torch
from torch import nn
import random

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,local_files_only=True)
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,local_files_only=True)

def lora_models_loader(lora_path='autodl-tmp/data/small',models_batch_size=4):
    models_list = os.listdir(os.path.join(lora_path, 'train'))
    random.shuffle(models_list)
    j = 0
    while j < len(models_list):
        yield models_list[j:j+models_batch_size]
        j += models_batch_size

def data_loader(data_path='autodl-tmp/data/eval_data.jsonl'):
    raw_data = []
    with jsonlines.open(data_path, 'r') as reader:
        for obj in reader:
            raw_data.append(obj)
    ref_data = {}
    eval_data = {}
    for item in raw_data:
        if item["lora_number"] == -1:
            ref_data[item["context"]] = item["evaluate"]
        else:
            if str(item["lora_number"]) in eval_data.keys():
                eval_data[str(item["lora_number"])].append({"context":item["context"],"evaluate":item["evaluate"]})
            else:
                eval_data[str(item["lora_number"])] = [{"context":item["context"],"evaluate":item["evaluate"]}]
    for k in eval_data.keys():
        for item in eval_data[k]:
            item["ref"] = ref_data[item["context"]]
    return eval_data

def data_iter(eval_data, lora_models_batch, data_batch_size=5):
    curr_data = []
    for lora in lora_models_batch:
        new_data = eval_data[lora][:]
        random.shuffle(new_data)
        curr_data.append(new_data)
    j = 0
    k = 0
    while k < len(curr_data[0]):
        yield curr_data[j][k:k+data_batch_size]
        if j < len(curr_data)-1:
            j += 1
        else:
            j = 0
            k += data_batch_size

def preprocess(example, tokenizer=tokenizer, config=config, max_seq_length=256):
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

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids.to("cuda"),
        "labels": labels.to("cuda"),
    }

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def main(data_path='autodl-tmp/data/eval_data.jsonl',lora_path='autodl-tmp/data/small',eval_path='autodl-tmp/data/pretrain',save_path='autodl-tmp/data/eval',models_batch_size=4,data_batch_size=5,epochs=3,learning_rate=0.0001):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path,'adapter_model.bin')):
        shutil.copy(os.path.join(eval_path,'adapter_model.bin'), save_path)
        shutil.copy(os.path.join(eval_path,'adapter_config.json'), save_path)
    eval_lora_params = torch.load(os.path.join(save_path,'adapter_model.bin'))
    eval_lora_params = {k:nn.Parameter(eval_lora_params[k].to("cuda")) for k in eval_lora_params.keys()}
    for k in eval_lora_params.keys():
        eval_lora_params[k].requires_grad = True
    eval_data=data_loader(data_path)
    optimizer = torch.optim.AdamW([eval_lora_params[key] for key in eval_lora_params], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        for i,lora_models_batch in enumerate(lora_models_loader(lora_path,models_batch_size)):
            print(f"Computing batch {i+1} : {lora_models_batch}...")
            # load all models in the current batch
            models = {}
            # first, models['-1'] is the base model
            models['-1'] = AutoModel.from_pretrained(
                "THUDM/chatglm-6b", trust_remote_code=True, device_map="auto",local_files_only=True
            ).half().cuda()
            models['-1'].gradient_checkpointing_enable()
            models['-1'].enable_input_require_grads()
            models['-1'].is_parallelizable = True
            models['-1'].model_parallel = True
            models['-1'].lm_head = CastOutputToFloat(models['-1'].lm_head)
            models['-1'].config.use_cache = (
                False  # silence the warnings. Please re-enable for inference!
            )
            for lora_id in lora_models_batch:
                models[lora_id] = AutoModel.from_pretrained(
                    "THUDM/chatglm-6b", trust_remote_code=True, device_map="auto",local_files_only=True
                ).half().cuda()
                models[lora_id].gradient_checkpointing_enable()
                models[lora_id].enable_input_require_grads()
                models[lora_id].is_parallelizable = True
                models[lora_id].model_parallel = True
                models[lora_id].lm_head = CastOutputToFloat(models[lora_id].lm_head)
                models[lora_id].config.use_cache = (
                    False  # silence the warnings. Please re-enable for inference!
                )
                models[lora_id] = PeftModel.from_pretrained(models[lora_id], lora_path + "/train/" + lora_id, lora_id)
                models[lora_id] = models[lora_id].merge_and_unload()                            
            for j,item_batch in enumerate(data_iter(eval_data, lora_models_batch, data_batch_size)):
                # for each batch of question-evaluate pairs
                # first apply the base model + eval lora, loss is the difference between output and 'ref'
                lora_id = lora_models_batch[j%models_batch_size]
                evals_without_lora = data_collator([preprocess({"context":item["context"], "target":item["ref"]}) for item in item_batch])
                evals_with_lora = data_collator([preprocess({"context":item["context"], "target":item["evaluate"]}) for item in item_batch])
                # init model
                if j==0:
                    models['-1'] = PeftModel.from_pretrained(models['-1'],save_path,"evaluate0")
                else:
                    models['-1'].load_adapter(save_path,adapter_name=f"evaluate{j}")
                models['-1'].set_adapter(f"evaluate{j}")
                for k in models['-1'].state_dict().keys():
                    model_attr = models['-1']
                    for attr in k.split('.'):
                        model_attr = getattr(model_attr, attr)
                    if 'lora' in k:
                        # k1 = k.replace(".evaluate.weight",".weight")
                        # # model.state_dict()[k] = eval_lora_params[k1]
                        # model_attr = eval_lora_params[k1]
                        model_attr.requires_grad_(True)
                        # model.state_dict()[k].requires_grad_(True)
                    else:
                        model_attr.requires_grad = False
                new_loss = 0.0
                # for w in range(data_batch_size):
                #     loss_print = model(input_ids=evals_without_lora["input_ids"][w:w+1,],labels=evals_without_lora["labels"][w:w+1,])
                #     print(loss_print)
                #     logits = model(evals_without_lora["input_ids"][w:w+1,],evals_without_lora["masks"][w:w+1,],evals_without_lora["labels"][w:w+1,]).loss['logits']
                #     new_loss += loss_fn(logits.view(-1, logits.shape[-1]), evals_without_lora['labels'][w:w+1,].view(-1))
                new_loss = models['-1'](input_ids=evals_without_lora["input_ids"],labels=evals_without_lora["labels"]).loss
                print('Loss: ', new_loss)
                new_loss.backward()
                for k in models['-1'].state_dict().keys():
                    model_attr = models['-1']
                    for attr in k.split('.'):
                        model_attr = getattr(model_attr, attr)
                    if 'lora' in k and f"evaluate{j}" in k:
                        k1 = k.replace(f".evaluate{j}.weight",".weight")
                        while 'base_model.model.base_model.model.' in k1:
                            k1 = k1.replace('base_model.model.base_model.model.','base_model.model.')
                        # model.state_dict()[k] = eval_lora_params[k1]
                        if eval_lora_params[k1].grad != None:
                            eval_lora_params[k1].grad += model_attr.grad
                        else:
                            eval_lora_params[k1].grad = model_attr.grad
                        # model.state_dict()[k].requires_grad_(True)
                    else:
                        model_attr.requires_grad = False
                if j>=1:
                    models['-1'].delete_adapter(f'evaluate{j-1}')
                new_loss = 0.0
                # then apply merge(the base model + answer lora) + eval lora, loss is the difference between output and 'evaluate'
                # init model, add the evaluate LoRA
                if j<models_batch_size:
                    models[lora_id] = PeftModel.from_pretrained(models[lora_id],save_path,f"evaluate{j}")
                else:
                    models[lora_id].load_adapter(save_path,adapter_name=f"evaluate{j}")
                models[lora_id].set_adapter(f"evaluate{j}")
                for k in models[lora_id].state_dict().keys():
                    model_attr = models[lora_id]
                    for attr in k.split('.'):
                        model_attr = getattr(model_attr, attr)
                    if 'lora' in k:
                        # k1 = k.replace(".evaluate.weight",".weight")
                        # model.state_dict()[k] = eval_lora_params[k1]
                        model_attr.requires_grad_(True)
                        # model_attr = eval_lora_params[k1]
                        # model.state_dict()[k].requires_grad_(True)
                    else:
                        model_attr.requires_grad = False
                new_loss = 0.0
                # for w in range(data_batch_size):
                #     logits = model(evals_without_lora["input_ids"][w:w+1,],evals_without_lora["masks"][w:w+1,]).loss['logits']
                #     new_loss += loss_fn(logits.view(-1, logits.shape[-1]), evals_without_lora['labels'][w:w+1,].view(-1))
                # new_loss.backward()
                new_loss = models[lora_id](input_ids=evals_with_lora["input_ids"],labels=evals_with_lora["labels"]).loss
                print('Loss: ', new_loss)
                new_loss.backward()
                for k in models[lora_id].state_dict().keys():
                    model_attr = models[lora_id]
                    for attr in k.split('.'):
                        model_attr = getattr(model_attr, attr)
                    if 'lora' in k and f"evaluate{j}" in k:
                        k1 = k.replace(f".evaluate{j}.weight",".weight")
                        while 'base_model.model.base_model.model.' in k1:
                            k1 = k1.replace('base_model.model.base_model.model.','base_model.model.')
                        # model.state_dict()[k] = eval_lora_params[k1]
                        if eval_lora_params[k1].grad != None:
                            eval_lora_params[k1].grad += model_attr.grad
                        else:
                            eval_lora_params[k1].grad = model_attr.grad
                        # model.state_dict()[k].requires_grad_(True)
                    else:
                        model_attr.requires_grad = False
                if j>=models_batch_size:
                    models[lora_id].delete_adapter(f'evaluate{j-models_batch_size}')
                new_loss = 0.0
                # update parameters by computed grads, after every LoRA model in current batch is evaluated once
                if (j+1)%models_batch_size==0:
                    optimizer.step()
                    scheduler.step()
                    for k in eval_lora_params.keys():
                        eval_lora_params[k].grad.zero_()
                    torch.save(eval_lora_params, os.path.join(save_path, 'adapter_model.bin'))
                    print(f"Step {(j+1)//models_batch_size} finished.")
            del models['-1']
            for lora_id in lora_models_batch:
                del models[lora_id]
            del models
            
main()