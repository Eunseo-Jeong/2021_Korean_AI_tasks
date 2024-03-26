"""
python COPA_submission.py
"""
from argparse import ArgumentParser
import json
from re import S
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

task = "COPA"
model_name = "tunib/electra-ko-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--target_epochs", default=1, type=int)
args = parser.parse_args()

model_path_name = f'tunib-electra-ko-base-{task}-{args.target_epochs}'

ckpt_name = f"model_save/{model_path_name}.pt"

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

eval_data = pd.read_csv("data/COPA/SKT_COPA_Test.tsv", delimiter="\t")
eval_text, eval_question, eval_1, eval_2 = (
    eval_data["sentence"].values,
    eval_data["question"].values,
    eval_data["1"].values,
    eval_data["2"].values,
)

dataset = [
    {"data": t + args.sep_token + q + args.sep_token + f + args.sep_token + s}
    for t, q, f, s in zip(eval_text, eval_question, eval_1, eval_2)
]

submission = []

with torch.no_grad():
    model.eval() 
    for idx, data in tqdm(enumerate(dataset)):
        text = data["data"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        classification_results = output.logits.argmax(-1)

        submission.append({"idx": idx+1,"label" : classification_results.item()+1})

    copa_dic = {"copa" : submission}

    with open(f"submission/{model_path_name}.json", 'w') as f:
        json.dump(copa_dic, f, indent= 4)