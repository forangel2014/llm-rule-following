import datasets
import json
from tqdm import tqdm
import string

variables = list(string.ascii_uppercase)
instruction = ""

dataset = datasets.load_from_disk(".")

for split in dataset.keys():
    nl_rule2id = {}
    fol_rule2id = {}
    rule_dict = {}
    inv_rule_dict = {}
    split_data = []
    data = dataset[split]
    for sample in tqdm(data):
        query = eval(sample['query'])
        relations = sample["f_comb"].split("-")
        if len(relations) > 2:
            a=1
        def parse_gender(string):
            parsed_dict = {}
            pairs = string.split(',')
            for pair in pairs:
                key, value = pair.split(':')
                parsed_dict[key] = value
            return parsed_dict
        genders = parse_gender(sample["genders"])
        
        nl_rule = "if "
        fol_rule = ""
        for idx in range(len(relations)):
            nl_rule += f"{variables[idx]} has a {relations[idx]} {variables[idx+1]}, "
            fol_rule += f"{relations[idx]}({variables[idx]}, {variables[idx+1]}) ∧ "
            #rule = f"if A has a {relations[0]} B, B has a {relations[1]} C, and A is {genders[query[0]]}, C is {genders[query[1]]}, then C is the {sample['target_text']} of A."
        nl_rule += f"and {variables[0]} is {genders[query[0]]}, {variables[idx+1]} is {genders[query[1]]}, then {variables[idx+1]} is the {sample['target_text']} of {variables[0]}."
        fol_rule += f"{genders[query[0]]}({variables[0]}) ^ {genders[query[1]]}({variables[idx+1]}) => {sample['target_text']}({variables[0]}, {variables[idx+1]})"
        
        rule = {"NL": nl_rule, "FOL": fol_rule}
        if rule["NL"] not in inv_rule_dict.keys():
            id_ = len(rule_dict)
            rule_dict[id_] = rule
            inv_rule_dict[rule["NL"]] = id_
        else:
            id_ = inv_rule_dict[rule["NL"]]
        
        split_data.append({
            "instruction": instruction,
            "input": f"{sample['story']}\nWho is {query[1]} to {query[0]}?",
            "output": sample["target_text"],
            "rule": [id_]
        })
        
    json.dump(split_data, open(f"{split}_data.json", "w"), indent=4)
    json.dump(rule_dict, open(f"{split}_rule.json", "w"), indent=4)