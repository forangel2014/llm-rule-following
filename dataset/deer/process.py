import pandas as pd
import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_gpt(messages, model="gpt-3.5-turbo", n=1, stream=False):
    success = False
    while not success:
        
        try:
            # use openai gpt
            response = openai.ChatCompletion.create(model=model, messages=messages, n=n)
            if not stream and n == 1:
                output = response.choices[0].message.content
                return output
            
        except (TimeoutError, openai.error.OpenAIError) as e:
            print(e)
            pass


#2. Include the rule as part of the question stem, placed below the question.
#Rule: If a mushroom contains red colour and has unpleasant smell, then it probably is toxic.
#Rule: If an animal eats meat, then it probably has a big size.


# meta_prompt = f"""Given an inferential rule and a text containing a scientific fact, where the text follows the given rule. Based on the fact and the rule, please help me create a multiple-choice question.
# Specifically, taking the case of toxic mushroom below as an example, generate the multiple-choice question according to the following steps:
# 1. Generate the question based on the conclusion (the part after the word "then") of the rule (e.g. toxic => what kind of mushroom is likely to be toxic?).
# 2. Remove any information from the given text that directly leads to the answer, while retaining relevant information related to the rule, which will serve as the correct answer (e.g. remove "...it is classified as a poisonous mushroom...").
# 3. Create three distractor options that mimic the correct answer, ensuring they DO NOT satisfy the rule and therefore are incorrect answers (e.g. the mushrooms in A, C and D are not toxic). 
# 4. Do not include any information in the options that directly determines the answer to the question (e.g. Do not mention any information about "toxic" in all options).

# For example:

# Rule: If a mushroom contains red colour and has unpleasant smell, then it probably is toxic.
# Text: Rubroboletus satanas, commonly known as Satan's bolete or the Devil's bolete, is a basidiomycete fungus of the bolete family (Boletaceae) and one of its most infamous members. It was known as Boletus satanas before its transfer to the new genus Rubroboletus in 2014, based on molecular phylogenetic data. Found in broad-leaved and mixed woodland in the warmer regions of Europe, it is classified as a poisonous mushroom, known to cause gastrointestinal symptoms of diarrhea and violent vomiting. However, reports of poisoning are rare, due to its striking appearance and at times putrid smell, which discourage casual experimentation.
# Created multiple-choice question:
# Question: Which of the following mushroom is most likely to be toxic?
# A. Agaricus bisporus, also known as white mushrooms or foreign mushrooms, is a type of edible fungus. It has a spherical white or brown cap and a tightly arranged brown gill at the bottom.
# B. Rubroboletus satanas, commonly known as Satan's bolete or the Devil's bolete, is a basidiomycete fungus of the bolete family (Boletaceae) and one of its most infamous members. It has striking appearance and at times putrid smell.
# C. Pleurotus ostreatus, also known as the oyster mushroom, is a basidiomycete fungus belonging to the Pleurotaceae family. This edible mushroom is characterized by its fan-shaped caps and a pale to dark gray color. Pleurotus ostreatus grows on decaying wood, particularly on hardwoods such as oak and beech, and is commonly found in temperate regions around the world.
# D. Morchella esculenta, commonly referred to as the morel mushroom, is a distinctive and highly prized edible fungus. Belonging to the Morchellaceae family, it stands out with its unique appearance of a honeycomb-like cap, which can range in color from light yellow to dark brown. Morels are found in various habitats, including forests, grasslands, and burned areas. 
# The correct answer is B.

# Now please help me create the following samples:

# Rule: If an animal eats meat, then it probably has a big size.
# Text: Generally, males vary in total length from 250 to 390 cm (98 to 154 in) and weigh between 90 and 300 kg (200 and 660 lb) with skull length ranging from 316 to 383 mm (12.4 to 15.1 in). Females vary in total length from 200 to 275 cm (79 to 108 in), weigh 65 to 167 kg (143 to 368 lb) with skull length ranging from 268 to 318 mm (10.6 to 12.5 in). In either sex, the tail represents about 0.6 to 1.1 m (2 ft 0 in to 3 ft 7 in) of the total length. The Bengal and Siberian tigers are amongst the tallest cats in shoulder height. They are also ranked among the biggest cats that have ever existed reaching weights of more than 300 kg (660 lb). The tigers of the Sunda islands are smaller and less heavy than tigers in mainland Asia, rarely exceeding 142 kg (313 lb) in weight. 
# Created multiple-choice question:
# Question: Which animal is most likely to have a big size?
# A. Kangaroos are commonly found in Australia. They feed on the leaves, bark, and tender buds of plants
# B. Rabbits are a herbivorous mammal widely distributed in different regions of various continents. They mainly feed on the tender leaves of grass, vegetables, and trees.
# C. Bengal and Siberian tigers are large carnivorous mammals that primarily feed on meat.
# D. Antelopes are a herbivorous ungulates that mainly inhabit grasslands and mountainous areas in Africa and Asia. They feed on grass, leaves, and tender buds.
# The correct answer is C.

# Rule: If a place is closer to the equator, then the rice grown in that place probably can be harvested with more times.
# Text: 
# """

meta_prompt = f"""Given an inferential rule, please help me create a multiple-choice question.
Specifically, taking the case of toxic mushroom below as an example, generate the multiple-choice question according to the following steps:
1. Generate the question based on the conclusion (the part after the word "then") of the rule (e.g. toxic => what kind of mushroom is likely to be toxic?).
2. Create a correct option that faithfully follows the given rule (e.g. Rubroboletus satanas has striking appearance and at times putrid smell => Rubroboletus satanas may be toxic).
3. Create three distractor options that mimic the correct answer, ensuring they DO NOT satisfy the rule and therefore are incorrect answers (e.g. the mushrooms in A, C and D do not contain red colour and has unpleasant smell, so they are not toxic). 
4. Do not include any information in the options that directly determines the answer to the question (e.g. Do not mention any information about "toxic" in all options).

For example:

Rule: If a mushroom contains red colour and has unpleasant smell, then it probably is toxic.
Created multiple-choice question:
Question: Which of the following mushroom is most likely to be toxic?
A. Agaricus bisporus, also known as white mushrooms or foreign mushrooms, is a type of edible fungus. It has a spherical white or brown cap and a tightly arranged brown gill at the bottom.
B. Rubroboletus satanas, commonly known as Satan's bolete or the Devil's bolete, is a basidiomycete fungus of the bolete family (Boletaceae) and one of its most infamous members. It has striking appearance and at times putrid smell.
C. Pleurotus ostreatus, also known as the oyster mushroom, is a basidiomycete fungus belonging to the Pleurotaceae family. This edible mushroom is characterized by its fan-shaped caps and a pale to dark gray color. Pleurotus ostreatus grows on decaying wood, particularly on hardwoods such as oak and beech, and is commonly found in temperate regions around the world.
D. Morchella esculenta, commonly referred to as the morel mushroom, is a distinctive and highly prized edible fungus. Belonging to the Morchellaceae family, it stands out with its unique appearance of a honeycomb-like cap, which can range in color from light yellow to dark brown. Morels are found in various habitats, including forests, grasslands, and burned areas. 
The correct answer is B.

Now please help me create the following samples:

Rule: If an animal eats meat, then it probably has a big size.
Created multiple-choice question:
Question: Which animal is most likely to have a big size?
A. Kangaroos are commonly found in Australia. They feed on the leaves, bark, and tender buds of plants
B. Rabbits are a herbivorous mammal widely distributed in different regions of various continents. They mainly feed on the tender leaves of grass, vegetables, and trees.
C. Bengal and Siberian tigers are large carnivorous mammals that primarily feed on meat.
D. Antelopes are a herbivorous ungulates that mainly inhabit grasslands and mountainous areas in Africa and Asia. They feed on grass, leaves, and tender buds.
The correct answer is C.

Rule: If a place is closer to the equator, then the rice grown in that place probably can be harvested with more times.
Created multiple-choice question:
Question: Which place is most likely to have rice that can be harvested more times?
A. Canada is a country located in North America. It has a diverse climate, ranging from temperate to subarctic, and is known for its production of wheat and canola.
B. Thailand is a Southeast Asian country known for its tropical climate and fertile plains. It is one of the world's largest exporters of rice and is renowned for its rice cultivation.
C. Norway is a country situated in Northern Europe. It has a cold and temperate climate, with limited agricultural land suitable for rice cultivation.
D. The North China Plain is a vast lowland region in northern China known for its fertile soil and favorable climate for rice cultivation.
The correct answer is B.

Rule: If a planet has a longer distance from its star, then it will have a longer orbitting period.
Created multiple-choice question:
Question: Which planet is most likely to have a longer orbiting period?
A. Mercury is the closest planet to the Sun in our solar system, with an average distance of about 36 million miles (58 million kilometers).
B. Venus is the second planet from the Sun and has an average distance of about 67 million miles (108 million kilometers).
C. Mars is the fourth planet from the Sun and has an average distance of about 142 million miles (228 million kilometers).
D. Jupiter is the largest planet in our solar system and has an average distance of about 484 million miles (778 million kilometers).
The correct answer is D.
"""


path = os.path.abspath(__file__)
dir_name = os.path.dirname(path)

split_data = []
rule2id = {}

def check_output(output):
    lines = output.split('\n')
    #question, rule, A, B, C, D = lines[:6]
    last_line = lines[-1].strip().lower()
    if "correct answer" in last_line:
        content = "\n".join(lines[:-1])
        answer = last_line.strip().strip(".").split(" ")[-1]
        return content, answer
    else:
        content = output
        print("*"*50)
        print(content)
        print("*"*50)
        print("please annotate answer: ")
        answer = input()
        return content, answer
        # 等待输入或执行其他操作


for split in ["test"]:
    data = pd.read_excel(f"{dir_name}/Hypothetical_Induction_{split}.xlsx").to_dict()
    questions = open(f"{dir_name}/questions_{split}.txt").readlines()
    questions = [question.split("->")[-1].strip() for question in questions]
    for id_ in data["id"]:
        #if "if" in data["rule"][id_].lower():
            rule = data['rule'][id_]
            print(rule)
            # print("please annotate question: ")
            # question = input()
            question = questions[id_]
            if question == "pass":
                continue
            question = question.capitalize()
            print(question)
            prompt = meta_prompt + f"Rule: {rule}\nCreated multiple-choice question:\nQuestion: {question}\n"
            for fact in ["fact 1.1"]:#, "fact 1.2", "fact 2.1", "fact 2.2", "fact 3.1", "fact 3.2"]:
                if data[fact][id_]:
                                
                    messages = [{'role': 'user', 'content': prompt}]
                    model = "gpt-3.5-turbo"
                    # model = "gpt-4-turbo"
                    # model = "gpt-4"
                    output = openai_gpt(messages=messages, model=model)

                    options, answer = check_output(output)
                    
                    print(options)
                    print(answer)
                    
                    split_data.append({
                        "instruction": "",
                        "input": question+options,
                        "output": answer,
                        "rule": [rule]
                    })
                    if rule not in rule2id.keys():
                        rule2id[rule] = len(rule2id)
    
    for data in split_data:
        data["rule"] = [rule2id[rule] for rule in data["rule"]]
    rule_dict = {}
    for rule, id_ in rule2id.items():
        rule_dict[id_] = {"NL": rule}
        
    json.dump(split_data, open(f"{dir_name}/{split}_data.json", "w"), indent=4)
    json.dump(rule_dict, open(f"{dir_name}/{split}_rule.json", "w"), indent=4)