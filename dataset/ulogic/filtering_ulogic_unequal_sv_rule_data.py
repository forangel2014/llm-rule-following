import json
from path import Path


drop_list = json.loads((Path(__file__).parent.parent/"ulogic-cf/not_equal_list.json").read_text())
drop_v_rules = {d['v_rule'] for d in drop_list}

filter_jsons = """rules.json
test_data.json
test_rule.json
train_data.json
train_rule.json"""

for json_name in filter_jsons.split("\n"):
    json_path = Path(__file__).parent/json_name
    data = json.loads(json_path.read_text())
    if "rule" in json_name:
        for key in list(data.keys()):
            if data[key]["NL"] in drop_v_rules:
                data.pop(key)
    elif "_data" in json_name:
        new_data = []
        for d in data:
            if d["rule in NL"] not in drop_v_rules:
                new_data.append(d)
        data = new_data
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1))

print("filtering over")
