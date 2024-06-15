import json
from pathlib import Path

raw_data = json.loads(Path("/home/zxc/Downloads/ULogic/Data/probing_subset.json").read_text())
format_FOL = "FOL"

need_update_file_names = ["rules.json", "test_rule.json", "train_rule.json"]
for file_name in need_update_file_names:
    goal_path = Path(__file__).parent/file_name
    original = json.loads(goal_path.read_text())
    for key, v in original.items():
        if format_FOL not in v:
            s_rule = raw_data[int(key)]['s_rule']
            conclusion, premise = s_rule.split(":-")
            predicates = [(predicate.strip().replace(");", ")") if ");" in predicate else predicate.strip()+")")\
                          for predicate in premise.split("),")]
            premise_fol = " ^ ".join(predicates)
            fol_rule = f"{premise_fol} => {conclusion}"
            v[format_FOL] = fol_rule

    goal_path.write_text(json.dumps(original, ensure_ascii=False, indent=1))

