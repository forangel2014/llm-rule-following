import json
from pathlib import Path

qas_names = ["qa-with-rule.json", "test_data.json", "train_data.json"]
rules = json.loads((Path(__file__).parent / "rules.json").read_text())

for qas_name in qas_names:
    qas_path = Path(__file__).parent / qas_name
    qas = json.loads(qas_path.read_text())
    nl_rule2rule_id = {rule["NL"]: rule_id for rule_id, rule in rules.items()}


    def any_startswith(text, words):
        for word in words:
            if text.startswith(word):
                return True
        return False


    def any_func(text, words, func):
        for word in words:
            if func(text, word):
                return True
        return False


    for qa in qas:
        positive_ = qa["positive"]
        v_rule = qa["rule in NL"]
        if v_rule in nl_rule2rule_id:
            rule_id = nl_rule2rule_id[v_rule]
            s_rule = rules[rule_id]["FOL"]
            hypothesis = s_rule.split("=>")[-1].strip()
            if qa["positive"]:
                if any_func(hypothesis, ["Cannot"], lambda x, y: y.lower() in x.lower()):
                    print(hypothesis, qa["input"].split("Question:")[-1].strip())
                    positive_ = False
            else:
                v_rule = qa["rule in NL"]
                if v_rule in nl_rule2rule_id:
                    rule_id = nl_rule2rule_id[v_rule]
                    s_rule = rules[rule_id]["FOL"]
                    hypothesis = s_rule.split("=>")[-1].strip()
                    if not any_startswith(hypothesis, ["Can", "Not"]):
                        # print(hypothesis, qa["input"].split("Question:")[-1].strip())
                        positive_ = True

        qa["positive_"] = positive_
    qas_path.write_text(json.dumps(qas, ensure_ascii=False, indent=1))
