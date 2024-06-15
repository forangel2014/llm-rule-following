from pathlib import Path
import json
import requests
from tqdm import tqdm
import openai
from collections import defaultdict
import random
import hashlib
import re

api_config = json.loads((Path(__file__).parent / "api_config.json").read_text())
openai.api_key = api_config["openai.api_key"]
openai.proxy = api_config["openai.proxy"]


def openai_gpt(messages, model="gpt-3.5-turbo", n=1, stream=False):
    # use openai gpt
    response = openai.ChatCompletion.create(model=model, messages=messages, n=n)
    if not stream and n == 1:
        output = response.choices[0].message.content
        return output


all_data = json.loads(Path("/home/zxc/Downloads/ULogic/Data/probing_subset.json").read_text())
save_rule_path = Path(__file__).parent / "rules.json"
rules = dict() if not save_rule_path.exists() else json.loads(save_rule_path.read_text())
if not save_rule_path.exists():
    for ix, item in tqdm(enumerate(all_data)):
        rule = item["v_rule"]
        instance = {"NL": rule}
        rules[ix] = instance
    # save
    save_rule_path.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
rule2id = {rule["NL"]: rule_id for rule_id, rule in rules.items()}

# gen qa dataset
save_qa_path = Path(__file__).parent / "qa-with-rule.json"
rule_qas = [] if not save_qa_path.exists() else json.loads(save_qa_path.read_text())
cache_v_rules = {instance["rule in NL"] for instance in rule_qas}

for item in tqdm(all_data):
    v_rule = item["v_rule"]
    # v_rule = "If Person X is allergic to Substance Z1 and Plant Y produces Substance Z1, then Person X cannot eat Plant Y."
    # v_rule = "If Person X lacks Skill Z1 and Tool Y is designed for Skill Z1, then Person X is not a repairman for Tool Y."
    premise, hypothesis = v_rule.split("then")
    premise = premise.replace("If", "").strip()
    hypothesis = hypothesis.strip()
    if v_rule in cache_v_rules:
        continue
    prompt = """Given premise and hypothesis,
please instantiate the Alphabetical Representation like A,B,C,X,Y,Z in both sentence to imaginary reasonable instance.
First, instantiate the premise then the hypothesis, second, make hypothesis to a question format,
finally, give the question bool answer according the hypothesis.
Please instantiate the premise with more extended lively detail.
While instantiate hypothesis and its question format concisely.
Output the whole result to a JSON like this:
{"premise_instantiated": "...", "hypothesis_instantiated": "...", "hypothesis_with_question_format": "..."}
Directly give out the JSON, no other explanation need.

Currently premise and hypothesis:
Premise:
{Premise}
Hypothesis:
{Hypothesis}""".replace("{Premise}", premise).replace("{Hypothesis}", hypothesis)
    messages = [{'role': 'user', 'content': prompt}]
    model = "gpt-4-turbo"
    output = openai_gpt(messages=messages, model=model)
    def parse_gpt4_json(s):
        try:
            return json.loads(s), s
        except:
            json_block = re.search(r'```json(.+?)```', s, re.DOTALL)
            if json_block:
                json_text = json_block.group(1)
                json_data = json.loads(json_text)
                return json_data, s
            else:
                return None, s
    info, s = parse_gpt4_json(output)
    assert v_rule in rule2id
    instance = {
        "instruction": "Giving a Context, please answer the Question.",
        "input": f"Context:\n{info['premise_instantiated']}\nQuestion:\n{info['hypothesis_with_question_format']}",
        "output": info["hypothesis_instantiated"],
        "rule": [rule2id[v_rule]],
        "rule in NL": v_rule,
        "depth": item["depth"],
        "length": item["length"],
        "positive": item["positive"],
        "label": item["label"],
        "original_human_prediction": item["original_human_prediction"],
        "flipped_human_prediction": item["flipped_human_prediction"],
        "domain": item["domain"],
        "structure": item["structure"]
    }
    rule_qas.append(instance)
    # save
    save_qa_path.write_text(json.dumps(rule_qas, ensure_ascii=False, indent=1))
    if not item["positive"]:
        # 答案是负向的
        print(instance)
print(f"num of qas with rule: {len(rule_qas)}")

# split test and train
import random
import shutil
# rule copy
shutil.copy(save_rule_path, save_qa_path.parent/"test_rule.json")
shutil.copy(save_rule_path, save_qa_path.parent/"train_rule.json")
# qas split
random.seed(0)
random.shuffle(rule_qas)
test_size = len(rule_qas)*4//5
save_qa_path_test = save_qa_path.parent/"test_data.json"
save_qa_path_train = save_qa_path.parent/"train_data.json"
save_qa_path_test.write_text(json.dumps(rule_qas[:test_size], ensure_ascii=False, indent=1))
save_qa_path_train.write_text(json.dumps(rule_qas[test_size:], ensure_ascii=False, indent=1))
