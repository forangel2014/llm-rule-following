from pathlib import Path
import json
import requests
from tqdm import tqdm
import openai
from collections import defaultdict
import random
import hashlib

api_config = json.loads((Path(__file__).parent / "api_config.json").read_text())
openai.api_key = api_config["openai.api_key"]
openai.proxy = api_config["openai.proxy"]


def openai_gpt(messages, model="gpt-3.5-turbo", n=1, stream=False):
    # use openai gpt
    response = openai.ChatCompletion.create(model=model, messages=messages, n=n)
    if not stream and n == 1:
        output = response.choices[0].message.content
        return output


all_theorem = json.loads(Path("/home/zxc/Downloads/TheoremQA-main/all_theorems.json").read_text())
save_rule_path = Path(__file__).parent / "rules.json"
rules = dict() if not save_rule_path.exists() else json.loads(save_rule_path.read_text())
rule_name2id = dict()
save_rule_name2id_path = Path(__file__).parent / "rule_name2id.json"
for ix, (theorem, theorem_content) in tqdm(enumerate(all_theorem.items())):
    theorem = theorem.lower()
    rule_name2id[theorem] = str(ix)
    if theorem in rules:
        continue
    prompt = f"""Please help me to translate the theorem to 'if ... then ...' format.
And keep information and computation detail as more as possible.
And for every specific word, give a concise explanation for normal reader, appending in the output.
Theorem info:
{theorem}: Content start:
{theorem_content}
Content end.
We define the (If_Then format and explanation) as a rule.Please give me the rule based on the theorem info.
Directly output the rule content only without any conclusion.
Rule:
"""
    messages = [{'role': 'user', 'content': prompt}]
    model = "gpt-4-turbo"
    output = openai_gpt(messages=messages, model=model)
    rule_in_if_then_format = output
    instance = {"NL": rule_in_if_then_format, "name": theorem}
    rules[rule_name2id[theorem]] = instance
    # save
    save_rule_path.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
    save_rule_name2id_path.write_text(json.dumps(rule_name2id, ensure_ascii=False, indent=1))

# get qa dataset
all_theorem_qas = json.loads(Path("/home/zxc/Downloads/TheoremQA-main/theoremqa_test.json").read_text())
rule_qas = []
for qa in all_theorem_qas:
    if qa["Picture"] is None and "list" not in qa["Answer_type"]:
        rule = rule_name2id[qa["theorem"]]
        assert rule in rules
        assert qa["Answer_type"] in {"integer", "float", "bool", "option"}
        input_ = qa["Question"].replace("(a)", "\nChoices:\na").replace("(b)", "b").replace("(c)", "c").replace("(d)", "d")
        if qa["Answer_type"] == "option":
            answer = qa["Answer"].replace("(", "").replace(")", "").strip()
        else:
            if qa["Answer_type"] in {"integer", "float"}:
                candidates = [-1, 0, 1, 2]
                if qa["Answer"] != 0:
                    candidates = [c * qa["Answer"] for c in candidates]
            elif qa["Answer_type"] == "bool":
                candidates = [True, False, "Unknown"]
            random.shuffle(candidates)
            options = list("abcd")
            candidates_str = "\nChoices:\n" + ". ".join(f"{options[ix]} {c}" for ix, c in enumerate(candidates))
            input_ += candidates_str
            answer = options[candidates.index(qa["Answer"])]
        instance = {
            "instruction": "",
            "input": input_,
            "output": answer,
            "rule": [rule],
            "Answer": qa["Answer"],
            "Answer_type": qa["Answer_type"],
            "subfield": qa["subfield"],
            "field": qa["field"]
        }
        rule_qas.append(instance)
print(f"num of qas with rule: {len(rule_qas)}")

# save
save_qa_path = Path(__file__).parent / "qa-with-rule.json"
save_qa_path.write_text(json.dumps(rule_qas, ensure_ascii=False, indent=1))
# split test and train
import shutil

# rule copy
shutil.copy(save_rule_path, save_qa_path.parent / "test_rule.json")
shutil.copy(save_rule_path, save_qa_path.parent / "train_rule.json")
# qas split
random.shuffle(rule_qas)
test_size = len(rule_qas) * 4 // 5
save_qa_path_test = save_qa_path.parent / "test_data.json"
save_qa_path_train = save_qa_path.parent / "train_data.json"
save_qa_path_test.write_text(json.dumps(rule_qas[:test_size], ensure_ascii=False, indent=1))
save_qa_path_train.write_text(json.dumps(rule_qas[test_size:], ensure_ascii=False, indent=1))
