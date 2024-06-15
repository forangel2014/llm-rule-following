import json
from pathlib import Path
import requests
from tqdm import tqdm
import openai
from collections import defaultdict
import random
import hashlib


api_config = json.loads((Path(__file__).parent/"api_config.json").read_text())
openai.api_key = api_config["openai.api_key"]
openai.proxy = api_config["openai.proxy"]


def openai_gpt(messages, model="gpt-3.5-turbo", n=1, stream=False):
    # use openai gpt
    response = openai.ChatCompletion.create(model=model, messages=messages, n=n)
    if not stream and n == 1:
        output = response.choices[0].message.content
        return output

raw_data = json.loads(Path("/home/zxc/Downloads/TheoremQA-main/all_theorems.json").read_text())
for key in list(raw_data.keys()):
    raw_data[key.lower()] = raw_data.pop(key)
save_rule_name2id_path = Path(__file__).parent / "rule_name2id.json"
rule_name2id = json.loads(save_rule_name2id_path.read_text())
key2rule_name = {id_: rule_name for rule_name, id_ in rule_name2id.items()}
format_FOL = "FOL"
goal_path = Path(__file__).parent/"rules.json"
original = json.loads(goal_path.read_text())

cache_key2fol = {key: v[format_FOL] for key, v in original.items() if format_FOL in v}
need_update_file_names = ["rules.json", "test_rule.json", "train_rule.json"]
for file_name in need_update_file_names:
    goal_path = Path(__file__).parent/file_name
    original = json.loads(goal_path.read_text())
    for key, v in tqdm(original.items()):
        if format_FOL not in v:
            if key in cache_key2fol:
                fol_rule = cache_key2fol[key]
            else:
                theorem = raw_data[key2rule_name[key]]
                prompt = f"""Please transform the following theorem to FOL(First Order Logic) format.
<Theorem>
{key}
{theorem}
</Theorem>
FOL:
"""
                messages = [{'role': 'user', 'content': prompt}]
                model = "gpt-4-turbo"
                output = openai_gpt(messages=messages, model=model)
                fol_rule = output
                cache_key2fol[key] = fol_rule
            v[format_FOL] = fol_rule
            goal_path.write_text(json.dumps(original, ensure_ascii=False, indent=1))

