from pathlib import Path
import json
import requests
from tqdm import tqdm
import openai
from collections import defaultdict
import random
import hashlib
import re


def predicate2counterfact_(predicate):
    if "CanNot" in predicate:
        return predicate.replace("CanNot", "Can")
    elif "Cannot" in predicate:
        return predicate.replace("Cannot", "Can")
    elif "Can" in predicate:
        return predicate.replace("Can", "CanNot")
    elif "Not" in predicate:
        return predicate.replace("Not", "")
    elif "HasSkills" in predicate:
        return "LacksSkills"
    elif "HasSkill" in predicate:
        return "LacksSkill"
    elif "LacksSkill" in predicate:
        return "HasSkill"
    elif "LacksSkills" in predicate:
        return "HasSkills"
    elif "Has" in predicate:
        return predicate.replace("Has", "HasNo")
    else:
        return "Not" + predicate
    return predicate


def predicate2normal_verbal(predicate):
    if predicate.startswith("LacksSkill"):
        verbal = "lacks the skill"
        if predicate.endswith("s"):
            verbal += "s"
    elif predicate == 'CannotSell':
        verbal = 'cannot be sold'
    elif predicate == "AppliesFor":
        verbal = "applies for"
    else:
        # split by uppercase
        verbal_words = re.findall(r'[A-Z][a-z]*', predicate)
        verbal = " ".join(word.lower() for word in verbal_words)
        verbal = verbal.replace("can not", "cannot")
        if "needs" not in verbal:
            verbal = verbal.replace("need", "needs")
    return verbal


def any_in(words, s):
    match = ""
    for word in words:
        if word in s:
            match = word
            break
    return match


def predicate_hypothesis2counterfact(predicate, hypothesis):
    predicate = "LivesInARegionWith" if predicate == "LivesInRegionWith" else predicate
    special_words = ["know", "affect", "need", "require"]

    if "CanNot" in predicate or "Cannot" in predicate or "cannot" in hypothesis:
        assert "cannot" in hypothesis
        return hypothesis.replace("cannot", "can")
    elif "Can" in predicate or "can" in hypothesis:
        assert "can" in hypothesis
        return hypothesis.replace("can", "cannot")
    elif "not" in hypothesis:
        return hypothesis.replace("not ", "")
    elif "is" in hypothesis:
        return hypothesis.replace(" is ", " is not ")
    elif "must" in hypothesis:
        return hypothesis.replace("must", "no need")
    elif any_in(special_words, hypothesis):
        match = any_in(special_words, hypothesis)
        return hypothesis.replace(match, f"not {match}")
    elif "HasSkills" in predicate:
        return "LacksSkills"
    elif "HasSkill" in predicate:
        return "LacksSkill"
    elif "Has" in predicate:
        return hypothesis.replace(" has ", " has no ")
    else:
        verbal_predicate = predicate2normal_verbal(predicate)
        counterfact = predicate2counterfact_(predicate)
        counterfact_verval = predicate2normal_verbal(counterfact)
        if verbal_predicate not in hypothesis:
            if verbal_predicate in hypothesis.lower():
                return hypothesis.lower().replace(verbal_predicate, counterfact_verval)
            else:
                replace_map = {
                    "lacks Skill": "has Skill"
                }
                for key, v in replace_map.items():
                    if key in hypothesis:
                        return hypothesis.replace(key, v)
                print(predicate, verbal_predicate, hypothesis)
                raise
        else:
            return hypothesis.replace(verbal_predicate, counterfact_verval)


all_data = json.loads(Path("/home/zxc/Downloads/ULogic/Data/probing_subset.json").read_text())
predicates = {d["s_rule"].split("(")[0] for d in all_data}
predicate2hypothesis = {d["s_rule"].split("(")[0]: d["v_rule"].split("then")[-1] for d in all_data}
predicate2counterfact = {predicate: predicate2counterfact_(predicate) for predicate in predicates}
predicate2verbal = {predicate: predicate2normal_verbal(predicate) for predicate in predicates}


# 修正
not_equal_list = []
all_data_goal = []
for d in all_data:
    s_rule = d["s_rule"]
    v_rule = d["v_rule"]
    predicate = d["s_rule"].split("(")[0]
    predicate_verbal = predicate2normal_verbal(predicate).replace(" the", "")
    hypothesis = d["v_rule"].split("then")[-1]
    if predicate_verbal not in hypothesis and not all(word in hypothesis.lower() for word in predicate_verbal.split(" ")):
        not_equal_list.append(d)
    else:
        all_data_goal.append(d)
not_equal_list_save_path = Path(__file__).parent / "not_equal_list.json"
not_equal_list_save_path.write_text(json.dumps(not_equal_list, ensure_ascii=False, indent=1))

save_rule_path = Path(__file__).parent / "rules.json"
save_rule_path_test_rule = Path(__file__).parent / "test_rule.json"
save_rule_path_train_rule = Path(__file__).parent / "train_rule.json"

rules = dict() if not save_rule_path.exists() else json.loads(save_rule_path.read_text())
if not rules:
    for ix, item in tqdm(enumerate(all_data)):
        if item in all_data_goal:
            rule = item["v_rule"]
            s_rule = item["s_rule"]
            predicate = s_rule.split("(")[0]
            hypothesis_ = rule.split("then")[-1]
            counterface_hypothesis = predicate_hypothesis2counterfact(predicate, hypothesis=hypothesis_)
            assert hypothesis_ in rule
            if "HasNoObligation" in counterface_hypothesis:
                counterface_hypothesis = predicate_hypothesis2counterfact(predicate, hypothesis=hypothesis_)
            rule_counterfact = rule.replace(hypothesis_, counterface_hypothesis)
            instance = {"NL": rule_counterfact}
            rules[ix] = instance
    # save
    save_rule_path.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
    save_rule_path_test_rule.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
    save_rule_path_train_rule.write_text(json.dumps(rules, ensure_ascii=False, indent=1))
rule2id = {rule["NL"]: rule_id for rule_id, rule in rules.items()}

# gen qa dataset
qas_names = ["qa-with-rule.json", "test_data.json", "train_data.json"]
for qas_name in qas_names:
    source_qa_path = Path(__file__).parent.parent/"ulogic"/qas_name
    source_qas = json.loads(source_qa_path.read_text())

    save_qa_path = Path(__file__).parent / qas_name
    rule_qas = []
    for item in tqdm(source_qas):
        if item["rule"][0] not in set(rule2id.values()):
            continue
        instance = dict(item)
        instance["positive_"] = not instance["positive_"]
        instance["output"] = str(instance["positive_"])
        old_rule_nl = item["rule in NL"]
        cf_rule_nl = rules[item["rule"][0]]["NL"]
        instance["rule in NL"] = cf_rule_nl
        assert old_rule_nl.split("then")[0].strip() == cf_rule_nl.split("then")[0].strip()
        rule_qas.append(instance)
    # save
    save_qa_path.write_text(json.dumps(rule_qas, ensure_ascii=False, indent=1))
    print(f"num of qas with rule: {len(rule_qas)}")

