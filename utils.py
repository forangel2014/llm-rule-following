import os
import re
import openai
import csv
import json
import random
from collections import Counter

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def list_subdirectories(directory):
    subdirectories = []
    try:
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                subdirectories.append(entry)
    except:
        pass
    return subdirectories

def list_files(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

def recursive_update(dictionary, info):
    for key, value in info.items():
        if key in dictionary:
            value_old = dictionary[key]
            dictionary[key] = recursive_update(value_old, value)
        else:
            dictionary[key] = value
        
    return dictionary

def find_all_indices(string, target):
    indices = []
    index = 0

    while index < len(string):
        if string[index] == target:
            indices.append(index)
        index += 1

    return indices

def first_part(text):
    dot_index = find_all_indices(text, ".")
    for index in dot_index:
        last_word = text[:index].split(" ")[-1]
        if len(last_word) >= 3 or last_word.isdigit():
            return text[:index] + "."
    return text + "."

def replace_first_quote_content(string):
    pattern = r"['\"](.*?)['\"]" 
    replace_string = re.sub(pattern, r'the input', string, count=1)
    return replace_string

def split_list(lst, num_parts):
    avg = len(lst) // num_parts  # 每个子列表的平均长度
    remainder = len(lst) % num_parts  # 余数

    result = []
    start = 0

    for i in range(num_parts):
        length = avg + 1 if i < remainder else avg
        result.append(lst[start:start+length])
        start += length

    return result

def split_data(data, ratio=0.1):
    
    random.shuffle(data)
    idx = round(len(data)*ratio)
    valid_data = data[:idx]
    train_data = data[idx:]
    
    return train_data, valid_data

def to_csv(filename, data):

    fieldnames = list(data[0].keys())

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in data:
            writer.writerow(row)

class Instance:
    
    def __init__(self, f, input_, output_, keys):
        
        self.keys = keys
        self.f = f
        self.input = input_
        self.output = output_

    def __str__(self):
        
        return f"{self.keys[0]}: {self.f}\n{self.keys[1]}: {self.input}\n{self.keys[2]}: {self.output}\n"

    def to_dict(self):
        
        return {self.keys[0]: self.instruction, self.keys[1]: self.input, self.keys[2]: self.output}

    def check(self):
        
        return self.f and self.input and self.output

def parse(text, keys):
    instances = []
    loop = True
    while loop:
        instance_value = []
        for key in keys:
            key += ": "
            if key in text and "\n" in text:
                _, text = text.split(key, 1)
                index = text.find("\n")
                if index != -1:
                    value = text[:index]
                    text = text[index+1:]
                else:
                    value = text
                    text = ""
                instance_value.append(value)
            else:
                loop = False
                break
        if loop:
            instance = Instance(*instance_value)
            instances.append(instance)
    return instances

def deduplicate_list_of_dicts(list_of_dicts):
    seen_instructions = set()
    deduplicated_list = []

    for d in list_of_dicts:
        instruction = d.get('instruction')
        if instruction is not None and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            deduplicated_list.append(d)

    return deduplicated_list

def prepare_data(filename, ratio):
    all_instances = json.load(open(f"./finetune/{filename}", 'r'))
    all_instances = [Instance(instance['instruction'], instance['input'], instance['output']) for instance in all_instances]
    all_instances = [instance for instance in all_instances if instance.check()]
    #all_instances = deduplicate_list_of_dicts(all_instances)

    # 设置CSV文件的列名
    fieldnames = ['input', 'target']
    fore_prompt = "I gave a friend an instruction and an input. The friend read the instruction and wrote an output for the input.\nHere is the input-output pair:\n"
    post_prompt = "The instruction was"

    instances = [{'input': fore_prompt + f"input: {instance.input}\noutput: {instance.output}\n" + post_prompt, 'target': instance.instruction} for instance in all_instances]
    #instances = [{'input': fore_prompt + f"input: {instance['input']}\noutput: {instance['output']}\n" + post_prompt, 'target': instance['instruction']} for instance in all_instances]

    random.shuffle(instances)
    num_instances = len(instances)
    num_train = round(ratio*num_instances)
    print(f"train samples: {num_train}")
    print(f"valid samples: {num_instances-num_train}")
    train_instances = instances[:num_train]
    valid_instances = instances[num_train:]

    # 指定CSV文件的路径
    train_file = './finetune/train.csv'
    valid_file = './finetune/valid.csv'

    # 使用'w'模式打开CSV文件
    with open(train_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入每一行数据
        for row in train_instances:
            writer.writerow(row)

    with open(valid_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入每一行数据
        for row in valid_instances:
            writer.writerow(row)

def sort_list_by_frequency(lst, log_probs):
    counter = Counter(lst)
    unique_list = list(set(lst))
    sorted_list = sorted(unique_list, key=lambda x: counter[x], reverse=True)
    sorted_log_probs = [log_probs[lst.index(elem)] for elem in sorted_list]
    final_list = [(sorted_list[i], sorted_log_probs[i]) for i in range(len(sorted_list))]
    return final_list

def generate_markdown_table(performance, order, tasks="all"):

    methods = order#list(list(performance.values())[0].keys())
    markdown_table = "| Task | " + " |".join(methods) + "\n"
    markdown_table += "| ---- | " + "-------- | "*len(methods) + "\n"

    for task in performance.keys():
        if task != "average" and (task in tasks or tasks == "all"):
            max_score_method = methods[0]
            max_score = performance[task][max_score_method]
            for method in methods:
                if performance[task][method] > max_score:
                    max_score = performance[task][method]
                    max_score_method = method
            row = f"| {task} "
            # for method, score in performance[task].items():
            #     row += f"| {score:.2f} "
            for method in methods:
                score = performance[task][method]*100
                if method == max_score_method:
                    row += f"| **{score:.2f}** "
                else:
                    row += f"| {score:.2f} "
            row += "|\n"
            markdown_table += row
    
    for task in performance.keys():
        if task == "average":
            max_score_method = methods[0]
            max_score = performance[task][max_score_method]
            for method in methods:
                if performance[task][method] > max_score:
                    max_score = performance[task][method]
                    max_score_method = method
            row = f"| {task} "
            # for method, score in performance[task].items():
            #     row += f"| {score:.2f} "
            for method in methods:
                score = performance[task][method]*100
                if method == max_score_method:
                    row += f"| **{score:.5f}** "
                else:
                    row += f"| {score:.5f} "
            row += "|\n"
            markdown_table += row
    
    return markdown_table

import torch
try:
    from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
        STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
except:
    from transformers import StoppingCriteria, StoppingCriteriaList
class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):

        self.token_id_list = token_id_list
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    # def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #     return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list
    
def find_combinations(lst):
    n = len(lst)
    combinations = []
    
    for i in range(n-1):
        for j in range(i+1, n):
                combination = [lst[i], lst[j]]
                combinations.append(combination)
    
    return combinations

import matplotlib.pyplot as plt
import numpy as np

def plot_radar(args):

    colors = [
        '#1f77b4', # Blue
        '#2ca02c', # Green
        '#d62728', # Red
        '#9467bd', # Purple
        '#e377c2', # Pink
        '#7f7f7f', # Gray
        '#ff7f0e', # Orange
        '#bcbd22', # Yellow-green
        '#17becf',  # Teal
        '#8c564b' # Brown   
    ]
    model_mapping = {
        "Llama-2-7b-chat-hf": "Llama-2-7b-chat",
        "Meta-Llama-3-8B": "Llama-3-8B",
        "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "Phi-3-small-8k-instruct": "Phi-3-small-8k-instruct",
        "Yi-1.5-6B-Chat": "Yi-1.5-6B-Chat"
    }
    models = list(os.listdir("./exp/clutrr/"))
    types = ["Executing Rules", "Triggering Rules", "Following\nFormal Rules", "Applying Rules", "Following Counterfactual Rules"]
    datasets = dict([(type, []) for type in types])
    
    #Executing Rules
    for model in models:
        performances = []
        for dataset_dir in os.listdir("./exp"):
            if "cf" not in dataset_dir:
                dataset_path = f"./exp/{dataset_dir}/{model}/golden_rule_NL"
                for shot in ["zero_shot", "few_shot"]:
                    for cot in ["cot", "no_cot"]:
                        result_path = f"{dataset_path}/{shot}/{cot}"
                        if os.path.isdir(result_path):
                            try:
                                result_file = f"{result_path}/result.json"
                                f = open(result_file, "r")
                                result_data = json.load(f)
                                performances.append(result_data)
                            except:
                                pass
        performance = np.mean(performances) if performances else 0
        datasets["Executing Rules"].append(performance)

    #Triggering Rules
    for model in models:
        performances = []
        for dataset_dir in os.listdir("./exp"):
            if "cf" not in dataset_dir:
                dataset_path = f"./exp/{dataset_dir}/{model}/all_rule_NL"
                for shot in ["zero_shot", "few_shot"]:
                    for cot in ["cot", "no_cot"]:
                        result_path = f"{dataset_path}/{shot}/{cot}"
                        if os.path.isdir(result_path):
                            try:
                                result_file = f"{result_path}/result.json"
                                f = open(result_file, "r")
                                result_data = json.load(f)
                                performances.append(result_data)
                            except:
                                pass
        performance = np.mean(performances) if performances else 0
        datasets["Triggering Rules"].append(performance)
        
    #Formal Rules
    for model in models:
        performances = []
        for dataset_dir in os.listdir("./exp"):
            if "cf" not in dataset_dir:
                for fol_dir in ["all_rule_FOL", "few_rule_FOL", "golden_rule_FOL"]:
                    dataset_path = f"./exp/{dataset_dir}/{model}/{fol_dir}"
                    for shot in ["zero_shot", "few_shot"]:
                        for cot in ["cot", "no_cot"]:
                            result_path = f"{dataset_path}/{shot}/{cot}"
                            if os.path.isdir(result_path):
                                try:
                                    result_file = f"{result_path}/result.json"
                                    f = open(result_file, "r")
                                    result_data = json.load(f)
                                    performances.append(result_data)
                                except:
                                    pass
        performance = np.mean(performances) if performances else 0
        datasets["Following\nFormal Rules"].append(performance)

    #Applying Rules
    for model in models:
        performances = []
        for dataset_dir in os.listdir("./exp"):
            if "cf" not in dataset_dir:
                for fol_dir in ["all_rule_NL", "few_rule_NL", "golden_rule_NL"]:
                    dataset_path = f"./exp/{dataset_dir}/{model}/{fol_dir}"
                    for shot in ["zero_shot", "few_shot"]:
                        for cot in ["cot"]:
                            result_path = f"{dataset_path}/{shot}/{cot}"
                            if os.path.isdir(result_path):
                                try:
                                    result_file = f"{result_path}/result.json"
                                    f = open(result_file, "r")
                                    result_data = json.load(f)
                                    performances.append(result_data)
                                except:
                                    pass
        performance = np.mean(performances) if performances else 0
        datasets["Applying Rules"].append(performance)     

    #Counterfactual Rules
    for model in models:
        performances = []
        for dataset_dir in os.listdir("./exp"):
            if "cf" in dataset_dir:
                for fol_dir in ["all_rule_NL", "few_rule_NL", "golden_rule_NL"]:
                    dataset_path = f"./exp/{dataset_dir}/{model}/{fol_dir}"
                    for shot in ["zero_shot", "few_shot"]:
                        for cot in ["cot"]:
                            result_path = f"{dataset_path}/{shot}/{cot}"
                            if os.path.isdir(result_path):
                                try:
                                    result_file = f"{result_path}/result.json"
                                    f = open(result_file, "r")
                                    result_data = json.load(f)
                                    performances.append(result_data)
                                except:
                                    pass
        performance = np.mean(performances) if performances else 0
        datasets["Following Counterfactual Rules"].append(performance)   
    
    avg = [np.mean([value[i] for value in datasets.values()]) for i in range(len(models))]
    models = [model_mapping[model.replace("vllm_", "")] for model in models]
    colors = {models[i]:colors[i] for i in range(len(models))}

    num_vars = len(datasets)

    # # sort the models based on the average score
    models = [x for _, x in sorted(zip(avg, models), reverse=True)]
    # # sort the datasets based on the average score

    # Normalize the datasets
    normalized_datasets = {}
    for key, values in datasets.items():
        values = [x for _, x in sorted(zip(avg, values), reverse=True)]
        min_val = min(values) 
        max_val = max(values)
        normalized_values = [ ((v - min_val) / (max_val - min_val) +0.03) * 100 for v in values]
        normalized_datasets[key] = normalized_values
        
        
    # Convert dictionary to list of tuples (key, values)
    labels, data = zip(*normalized_datasets.items())


    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(figsize=(8.5, 6), subplot_kw=dict(polar=True)) # 减小画布宽度

    # 调整子图的位置和大小
    fig.subplots_adjust(left=-0.075, right=1.0, top=0.9, bottom=0.1)

    # Plot for the first dataset
    for i, model in enumerate(models):
        values = [data[j][i] for j in range(num_vars)]
        values += values[:1]
        axs.fill(angles, values, color=colors[model], alpha=0.25)
        axs.plot(angles, values, '*-', color=colors[model], linewidth=2, label='#'+str(i+1)+'.'+ model)

    axs.set_xticks(angles[:-1])
    axs.set_yticklabels([])
    xticks = axs.set_xticklabels(labels, size=15)
    axs.legend(loc='upper center', bbox_to_anchor=(1.15, 1.01), prop={'size': 8})

    for label, angle in zip(xticks, angles):
        if angle >= 0 and angle < np.pi / 2:
            label.set_horizontalalignment('left')
            label.set_verticalalignment('bottom')
        elif angle >= np.pi / 2 and angle < np.pi:
            label.set_horizontalalignment('right')
            label.set_verticalalignment('bottom')
        elif angle >= np.pi and angle < 3 * np.pi / 2:
            label.set_horizontalalignment('right')
            label.set_verticalalignment('top')
        else:
            label.set_horizontalalignment('left')
            label.set_verticalalignment('top')
        label.set_rotation(np.degrees(angle) - 90)

    #axs.set_title("The Rule-Following Capabilities of Different LLMs", size=18, pad=10, loc='center')

    plt.savefig("./exp/radar.pdf")
    
def plot_barchart(data, name):
    datasets = list(data.keys())
    models = []
    for dataset in datasets:
        for model in list(data[dataset].keys()):
            if model not in models:
                models.append(model)
    num_models = len(models)
    num_datasets = len(datasets)
    colors = [
        '#1f77b4', # Blue
        '#2ca02c', # Green
        '#d62728', # Red
        '#9467bd', # Purple
        '#e377c2', # Pink
        '#7f7f7f', # Gray
        '#ff7f0e', # Orange
        '#bcbd22', # Yellow-green
        '#17becf',  # Teal
        '#8c564b' # Brown   
    ]
    model_mapping = {
        "Llama-2-7b-chat-hf": "Llama-2-7b-chat",
        "Meta-Llama-3-8B": "Llama-3-8B",
        "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "Phi-3-small-8k-instruct": "Phi-3-small-8k-instruct",
        "Yi-1.5-6B-Chat": "Yi-1.5-6B-Chat"
    }
    dataset_mapping = {
        "clutrr": "CLUTRR",
        "salad": "SALAD",
        "theoremQA": "TheoremQA",
        "deer": "DEER",
        "ulogic": "ULogic",
        "law": "CAIL2018"
    }
    keywords_mapping = None
    if name == "exp1":
        keywords_mapping = {
            "no_rule": "No Rule",
            "all_rule": "All Rule",
            "few_rule": "Few Rule",
            "golden_rule": "Golden Rule"
        }
    elif name == "exp2":
        keywords_mapping = {
            "FOL": "FOL",
            "NL": "NL",
        }
    elif name == "exp3":
        keywords_mapping = {
            "no_cot": "w/o CoT",
            "cot": "w/ CoT"
        }
    elif name == "exp4":
        keywords_mapping = {
            "counterfactual": "counterfactual",
            "factual": "factual",
        }
    elif name == "exp5":
        keywords_mapping = {
            "fire_error_rate": "Triggering Error",
            "execution_error_rate": "Execution Error",
        }
    xlabels_mapping = {
        "no_rule": "No Rule",
        "all_rule": "All Rule",
        "few_rule": "Few Rule",
        "golden_rule": "Golden Rule",
        " ": " "
    }
    fig, axs = plt.subplots(num_datasets, num_models, figsize=(16, 6))

    for i, dataset in enumerate(datasets):
        
        for j, model in enumerate(models):
            
            try:
            
                model_data = data[dataset][model]
                subplot_row = i
                subplot_col = j

                ax = axs[subplot_row, subplot_col]

                x_labels = list(model_data.keys())
                x_labels = [xlabels_mapping[x] for x in x_labels]
                all_x = np.arange(len(x_labels))
                ax.set_xticks(all_x)
                ax.set_xticklabels(list(x_labels), fontsize="small")  # Set x-axis tick label as 'k' value
                width = 0.35
                
                compare_types = list(keywords_mapping.keys()) if keywords_mapping else list(list(model_data.values())[0].keys())
                n = len(compare_types)
                
                for k in range(n):
                    
                    type = compare_types[k]
                    label = keywords_mapping[type] if keywords_mapping else type
                    type_values = []
                    for rule_data in model_data.values():
                        if type in rule_data.keys():
                            type_values.append(rule_data[type])
                        else:
                            type_values.append(0)

                    rects = ax.bar(all_x + width * (k-(n-1)/2), type_values, width, label=label, color=colors[k])
                        
                    def autolabel(rects):
                        for rect in rects:
                            height = rect.get_height()
                            ax.annotate(f"{height}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                                        textcoords="offset points", ha="center", va="bottom", fontsize="small")  # Reduce annotation font size

                    #autolabel(rects)
            
            except:
                pass


    # Add labels for model columns
    for ax, model in zip(axs[0], models):
        ax.set_title(model_mapping[model.replace("vllm_", "")], fontsize="medium", fontweight="bold", pad=10)

    # Add labels for datasets
    for ax, dataset in zip(axs[:, 0], datasets):
        ax.set_ylabel(dataset_mapping[dataset], fontsize="medium", fontweight="bold")

    # Create a single legend for the entire figure
    all_handles = []
    all_labels = []
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            handles, labels = axs[i, j].get_legend_handles_labels()
            if len(labels) > len(all_labels):
                all_handles = handles
                all_labels = labels
    fig.legend(all_handles, all_labels, loc="upper right", fontsize="small")
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

    plt.savefig(f"./exp/{name}.pdf")