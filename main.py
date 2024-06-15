import argparse
from multiprocessing import process
import os
import json
import httpx
import matplotlib.pyplot as plt
import numpy as np
from lm import LLM, RCD
import openai
from openai import OpenAI, OpenAIError
from processor import TaskDataProcessor
from utils import mkdir, plot_radar, plot_barchart, recursive_update
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings
import re, time

def generate_directory_table(args):
    if args.keywords:
        if args.keywords == "exp1":
            keywords = ["zero_shot", "no_cot"]
            args.ban_keywords = "analysis,FOL"
        elif args.keywords == "exp2":
            keywords = ["zero_shot", "no_cot"]
            args.ban_keywords = "analysis,no_rule,golden_rule"
        elif args.keywords == "exp3":
            keywords = ["few_shot", "NL"]
            args.ban_keywords = "no_rule,cf,analysis"
        elif args.keywords == "exp4":
            keywords = ["few_shot", "NL"]
            args.ban_keywords = "no_cot"
        elif args.keywords == "exp5":
            keywords = ["analysis", "few_shot", "NL"]
            args.ban_keywords = "no_cot"
        else:
            keywords = args.keywords.split(",")
            keywords = [keyword.strip() for keyword in keywords]
    if args.ban_keywords:
        ban_keywords = args.ban_keywords.split(",")
        ban_keywords = [keyword.strip() for keyword in ban_keywords]

    data = {}
    table = "| Dataset | Model | Rule Setting | Shot | CoT | Result |\n| --- | --- | --- | --- | --- | --- |\n"
    for dataset_dir in os.listdir("./exp"):
        dataset_path = os.path.join("./exp", dataset_dir)
        if os.path.isdir(dataset_path):
            for model_dir in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model_dir)
                if os.path.isdir(model_path):
                    for rule_dir in os.listdir(model_path):
                        rule_path = os.path.join(model_path, rule_dir)
                        if os.path.isdir(rule_path):
                            for shot_dir in os.listdir(rule_path):
                                shot_path = os.path.join(rule_path, shot_dir)
                                if os.path.isdir(shot_path):
                                    for cot_dir in os.listdir(shot_path):
                                        cot_path = os.path.join(shot_path, cot_dir)
                                        if os.path.isdir(cot_path):
                                            result_file = os.path.join(cot_path, "result.json")
                                            if os.path.isfile(result_file):
                                                try:
                                                    f = open(result_file, "r")
                                                    result_data = json.load(f)
                                                    if type(result_data) == float:
                                                        result_data = round(result_data * 100, 2)
                                                        info = f"| {dataset_dir} | {model_dir} | {rule_dir} | {shot_dir} | {cot_dir} | {result_data} |\n"
                                                    else:
                                                        assert type(result_data) == dict
                                                        result_data = {key: round(value * 100, 2) for key, value in result_data.items()}
                                                        info = f"| {dataset_dir} | {model_dir} | {rule_dir} | {shot_dir} | {cot_dir} | {str(result_data)} |\n"
                                                    include_keywords = True
                                                    if args.keywords:
                                                        for keyword in keywords:
                                                            if keyword not in info:
                                                                include_keywords = False
                                                                break
                                                    if args.ban_keywords:
                                                        for keyword in ban_keywords:
                                                            if keyword in info:
                                                                include_keywords = False
                                                                break
                                                    if include_keywords:
                                                        table += info
                                                        if args.keywords:
                                                            if args.keywords == "exp1":
                                                                rule_num, rule_type = rule_dir.rsplit("_", 1)
                                                                rule_num = "no_rule" if rule_num == "no" else rule_num
                                                                data = recursive_update(data, {dataset_dir: {model_dir: {" ": {rule_num: result_data}}}})
                                                            elif args.keywords == "exp2":
                                                                rule_num, rule_type = rule_dir.rsplit("_", 1)
                                                                data = recursive_update(data, {dataset_dir: {model_dir: {rule_num: {rule_type: result_data}}}})
                                                            elif args.keywords == "exp3":
                                                                rule_num, rule_type = rule_dir.rsplit("_", 1)
                                                                data = recursive_update(data, {dataset_dir: {model_dir: {rule_num: {cot_dir: result_data}}}})
                                                            elif args.keywords == "exp4":
                                                                rule_num, rule_type = rule_dir.rsplit("_", 1)
                                                                cf = None
                                                                clean_datasetdir = dataset_dir.replace("-cf", "")
                                                                all_datasets = os.listdir("./exp")
                                                                if "cf" in dataset_dir and clean_datasetdir in all_datasets:
                                                                    cf = "counterfactual"
                                                                elif f"{dataset_dir}-cf" in all_datasets:
                                                                    cf = "factual"
                                                                if type(result_data) == float and cf:
                                                                    data = recursive_update(data, {clean_datasetdir: {model_dir: {rule_num: {cf: result_data}}}})
                                                            elif args.keywords == "exp5":
                                                                clean_datasetdir = dataset_dir.replace("-cf", "")
                                                                result_data.pop("accuracy")
                                                                data = recursive_update(data, {clean_datasetdir: {model_dir: {" ": result_data}}})
                                                except Exception as e:
                                                    print(f"when processing result file {result_file}, encounter error {e}")
                                                    pass
    
    name = args.results_name if args.results_name else "results"
    g = open(f"./exp/{name}.md", "w")
    g.writelines(table)

    if args.keywords:
        if args.keywords in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
            plot_barchart(data, args.keywords)
    return table

def analysis_mechanism(args, processor, llm):

    def att_viz(input_text, tokenizer, model, name):
        
        case_dir = f"{args.exp_dir}/{name}"
        mkdir(case_dir)
        
        t = open(f"{case_dir}/{name}.txt", "w")
        t.writelines(input_text)
        
        for layer in [31]:#range(32):
            head = 0
            inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
            outputs = model(inputs, output_attentions=True)  # Run model
            attention = outputs[-1]  # Retrieve attention from model outputs
            tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
            #views = model_view(attention, tokens)  # Display model view
            layer_attention = attention[layer][0][head].detach().numpy()
            
            plt.figure(figsize=(10, 10), dpi=1000)
            plt.imshow(layer_attention, cmap='hot', interpolation='nearest')
            plt.xticks(np.arange(len(tokens)), tokens, fontsize=4, rotation=90)
            plt.yticks(np.arange(len(tokens)), tokens, fontsize=4)
            plt.colorbar()
            title = f'Layer {layer} Head 1 {name}'
            plt.title(title)
            plt.savefig(f"{case_dir}/{title}.png")

        # specified_token_index = 52

        # # 提取指定token与其他所有token的注意力值
        # specified_token_attention = layer_attention[:specified_token_index+1, specified_token_index]

        # # 绘制指定token与其他所有token的注意力矩阵
        # plt.imshow(specified_token_attention.reshape(1, -1), cmap='hot', aspect='auto')
        # plt.xticks(np.arange(specified_token_index+1), tokens[:specified_token_index+1], rotation=90)
        # plt.yticks([])
        # plt.colorbar()
        # title = f'Layer 1 Head 1, Attention to "{tokens[52]}" from Other Tokens'
        # plt.title(title)
        # plt.savefig(f"{case_dir}/{title}.png")

    
    for id_, samples in enumerate(tqdm(processor.test_dataloader)):
        
        processor.args.rule_setting = "no_rule"
        
        prompts_no_rule = processor.prompt(samples=samples)
        
        answers_no_rule = llm(prompts_no_rule, stop="\n")
        
        scores_no_rule = processor.eval_answer(samples=samples, answers=answers_no_rule)
        
        # processor.args.rule_setting = "golden_rule"
        
        # prompts_rule = processor.prompt(samples=samples)
        
        # answers_rule = llm(prompts_rule, stop="\n")
        
        # scores_rule = processor.eval_answer(samples=samples, answers=answers_rule)
        
        for i in tqdm(range(len(samples))):
            
            # att_viz(prompts_rule[i] + answers_rule[i], llm.tokenizer, llm.model, name=f"id={i}, no_rule={scores_no_rule[i]}, Golden_rule={scores_rule[i]}")
            att_viz(prompts_no_rule[i] + answers_no_rule[i], llm.tokenizer, llm.model, name=f"id={i}, score={scores_no_rule[i]}")
        
        break
    
def analysis_behaviour(args, processor, llm):

    t = open(f"{args.exp_dir}/test.txt", "w",encoding="utf-8")

    fire_errors = 0
    #binding_errors = 0
    execution_errors = 0
    total_score = num_test = 0

    for id_, samples in enumerate(tqdm(processor.test_dataloader)):
        answers=[]
        prompts=[]
        infos=[]
        if args.model_name_or_path.lower() in ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:

            for sample in samples:
                success = False
                while not success:
                    try:
                        prompts_infos = processor.prompt(samples=[sample])
                        answer = llm(prompts_infos[0][0])
                        answers.append(answer)
                        prompts.append(prompts_infos[0][0])
                        infos.append(prompts_infos[0][1])
                        success = True
                    except (TimeoutError, OpenAIError) as e:
                        if re.findall(".* Please try again in (.*)ms. Visit.*",e.__str__()):
                            sleep_time = re.findall(".* Please try again in (.*)ms. Visit.*", e.__str__())
                            sleep_time = float(sleep_time.pop())
                            print(f"token限制访问，需延时{sleep_time}ms后恢复访问.........")
                            time.sleep(sleep_time/1000)
                        elif re.findall(".* Please try again in (.*)s. Visit.*",e.__str__()):
                            sleep_time =re.findall(".* Please try again in (.*)s. Visit.*",e.__str__())
                            sleep_time=float(sleep_time.pop())
                            print(f"token限制访问，需延时{sleep_time}s后恢复访问.........")
                            time.sleep(sleep_time)
                        else:
                            if re.findall("This model's maximum context length is (\d+) tokens.*",e.__str__())  :
                                print("token 超长 : ",end=" ")
                                print(e)
                                success = True

        else:
            prompts_infos = processor.prompt(samples=samples)
            prompts = [p[0] for p in prompts_infos]
            infos = [p[1] for p in prompts_infos]
            success = False
            while not success:
                try:
                    answers = llm(prompts)
                    success = True
                # modified for newest openai pkg
                except (TimeoutError, OpenAIError) as e:
                    print(e)

        for i in range(len(answers)):

            score, fire_error, execution_error = processor.parse(samples[i], answers[i], infos[i])
            
            if score >= 0:

                fire_errors += fire_error
                #binding_errors += binding_error
                execution_errors += execution_error
                total_score += score
                num_test += 1
                num_failure = num_test - total_score
                num_failure = 1 if num_failure == 0 else num_failure

                t.writelines("#"*50+"\n")
                t.writelines("\nPROMPT:\n")
                t.writelines(prompts[i])
                t.writelines("\n\nGENERATE RESPONSE:\n")
                t.writelines(answers[i])
                t.writelines("\n\nGROUND TRUTH:\n")
                t.writelines(str(samples[i]["output"]))
                t.writelines("\n\nSCORE:\n")
                t.writelines(str(score))
                t.writelines("\n"+"#"*50+"\n")
                result = {"fire_error_rate": fire_errors/num_failure, "execution_error_rate": execution_errors/num_failure, "accuracy": total_score/num_test}
                print(result)
                t.writelines(json.dumps(result))
            
            # if id_ == 0:
            #     att_viz(prompts[i] + answers[i], llm.tokenizer, llm.model, name=f"id={i}, fire_error={fire_error}, binding_error={binding_error}, execution_error={execution_error}")
        
    f = open(f"{args.exp_dir}/result.json", "w")
    json.dump(result, f, indent=4)

def train(args, processor, llm):

    total_score = num_test = 0
    t = open(f"{args.exp_dir}/train.txt", "w",encoding="utf-8")

    success_samples = []
    failure_samples = []

    processor.test_dataloader = processor.train_dataloader
    processor.test_rule = processor.train_rule

    for id_, samples in enumerate(tqdm(processor.train_dataloader)):
        # 修改gpt调用方法
        answers = []
        prompts = []
        if args.model_name_or_path.lower() in  ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:

            for sample in samples:
                success = False
                all_number = 20
                while not success:
                    try:
                        prompt = processor.prompt(samples=[sample],all_number=all_number)

                        answer = llm(prompt)
                        answers.append(answer.pop())
                        prompts.append(prompt)
                        success = True
                    except (TimeoutError, openai.OpenAIError) as e:

                        if re.findall(".* Please try again in (.*)ms. Visit.*",e.__str__()):
                            sleep_time = re.findall(".* Please try again in (.*)ms. Visit.*", e.__str__())
                            sleep_time = float(sleep_time.pop())
                            print(f"token限制访问，需延时{sleep_time}ms后恢复访问.........")
                            time.sleep(sleep_time/1000)
                        elif re.findall(".* Please try again in (.*)s. Visit.*",e.__str__()):
                            sleep_time =re.findall(".* Please try again in (.*)s. Visit.*",e.__str__())
                            sleep_time=float(sleep_time.pop())
                            print(f"token限制访问，需延时{sleep_time}s后恢复访问.........")
                            time.sleep(sleep_time)
                        else:
                            print(e)
                            all_number-=1
        else:
            prompts = processor.prompt(samples=samples)
            success = False
            while not success:
                try:
                    answers = llm(prompts)
                    success = True
                # modified for newest openai pkg
                except (TimeoutError) as e:
                    print(e)

        scores = processor.eval_answer(samples=samples, answers=answers)

        total_score += sum(scores)
        num_test += len(scores)

        for i in range(len(samples)):
            if scores[i]:
                success_samples.append([samples[i], prompts[i], answers[i]])
            else:
                failure_samples.append([samples[i], prompts[i], answers[i]])

    json.dump(success_samples, open(f"{args.exp_dir}/success.json", "w"), indent=4)
    json.dump(failure_samples, open(f"{args.exp_dir}/failure.json", "w"), indent=4)

def test_rcd(args, processor, rcd):

    total_score = num_test = 0
    t = open(f"{args.exp_dir}/test_rcd_{args.alpha}.txt", "w",encoding="utf-8")

    for id_, samples in enumerate(tqdm(processor.test_dataloader)):
        # 修改gpt调用方法
        answers = []
        prompts = []
        if args.model_name_or_path.lower() in ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
            raise ValueError("model must be white-box LLM when using RCD")
        else:
            prompts = processor.prompt(samples=samples)
            success = False
            while not success:
                try:
                    answers = rcd.generate(prompts, alpha=args.alpha, max_length=args.max_token)
                    success = True
                # modified for newest openai pkg
                except OpenAIError as e:
                    print("OpenAI API Error:", e)
                except KeyboardInterrupt:
                    raise

        scores = processor.eval_answer(samples=samples, answers=answers)
        valid_scores = [score for score in scores if score >= 0]

        total_score += sum(valid_scores)
        num_test += len(valid_scores)

        for i in range(len(samples)):
            t.writelines("#"*50+"\n")
            t.writelines("\nPROMPT:\n")
            t.writelines("\n--------Positive--------:\n")
            t.writelines(prompts[i][0])
            t.writelines("\n--------Negative--------:\n")
            t.writelines(prompts[i][1])
            t.writelines("\n\nGENERATE RESPONSE:\n")
            t.writelines(answers[i])
            t.writelines("\n\nGROUND TRUTH:\n")
            t.writelines(str(samples[i]["output"]))
            t.writelines("\n\nSCORE:\n")
            t.writelines(str(scores[i]))
            t.writelines("\n"+"#"*50+"\n")

        print(total_score, num_test, total_score/num_test)

    f = open(f"{args.exp_dir}/result_rcd_{args.alpha}.json", "w",encoding="utf-8")
    f.write(str(total_score/num_test))
    f.close()

def test(args, processor, llm):

    total_score = num_test = 0
    t = open(f"{args.exp_dir}/test.txt", "w",encoding="utf-8")
    str_test = ""

    for id_, samples in enumerate(tqdm(processor.test_dataloader)):
        # 修改gpt调用方法
        answers = []
        prompts = []
        new_samples = []
        for sample in samples:
            success = False
            all_number = 20
            while not success:
                try:
                    prompt = processor.prompt(samples=[sample], all_number=all_number)

                    answer = llm(prompt)
                    answers.append(answer.pop())
                    prompts.append(prompt)
                    new_samples.append(sample)
                    success = True
                except OpenAIError as e:

                    if re.findall(".* Please try again in (.*)ms. Visit.*",e.__str__()):
                        sleep_time = re.findall(".* Please try again in (.*)ms. Visit.*", e.__str__())
                        sleep_time = float(sleep_time.pop())
                        print(f"token限制访问，需延时{sleep_time}ms后恢复访问.........")
                        time.sleep(sleep_time/1000)
                    elif re.findall(".* Please try again in (.*)s. Visit.*",e.__str__()):
                        sleep_time =re.findall(".* Please try again in (.*)s. Visit.*",e.__str__())
                        sleep_time=float(sleep_time.pop())
                        print(f"token限制访问，需延时{sleep_time}s后恢复访问.........")
                        time.sleep(sleep_time)
                    else:
                        if re.findall("This model's maximum context length is (\d+) tokens.*",e.__str__()) and args.rule_setting.lower()!="all_rule" or  all_number<=3 :
                            print("token 超长.....")
                            break
                        print(e)
                        all_number-=1
        samples=new_samples

        scores = processor.eval_answer(samples=samples, answers=answers)
        valid_scores = [score for score in scores if score >= 0]

        total_score += sum(valid_scores)
        num_test += len(valid_scores)
        
        for i in range(len(samples)):
            t.writelines("#"*50+"\n")
            t.writelines("\nPROMPT:\n")
            t.writelines(prompts[i])
            t.writelines("\n\nGENERATE RESPONSE:\n")
            t.writelines(answers[i])
            t.writelines("\n\nGROUND TRUTH:\n")
            t.writelines(str(samples[i]["output"]))
            t.writelines("\n\nSCORE:\n")
            t.writelines(str(scores[i]))
            t.writelines("\n"+"#"*50+"\n")
        
        print(total_score, num_test, total_score/num_test)
    
    f = open(f"{args.exp_dir}/result.json", "w",encoding="utf-8")
    f.write(str(total_score/num_test))
    f.flush()
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="law", help='name of dataset.')
    parser.add_argument('--rule_type', type=str, default="NL", help='type of rule used.[NL , FOL]')
    parser.add_argument('--batchsize', type=int, default=16, help='input batchsize.')
    parser.add_argument('--max_token', type=int, default=500, help='max number of tokens to generate.')
    parser.add_argument('--rule_setting', type=str, default="few_rule", help='[no_rule , golden_rule , few_rule , all_rule]')
    parser.add_argument('--ignore_exist', action="store_true", default=False, help='whether show results')
    parser.add_argument('--few_shot', action="store_true", default=False, help='whether do behaviour analysis.[true , false]')
    parser.add_argument('--cot', action="store_true", default=False, help='whether do behaviour analysis.[true , false]')
    parser.add_argument('--analysis_behaviour', action="store_true", default=False, help='whether do behaviour analysis')
    parser.add_argument('--analysis_mechanism', action="store_true", default=False, help='whether do mechanism analysis')
    parser.add_argument('--training', action="store_true", default=False, help='whether do mechanism analysis')
    parser.add_argument('--use_rcd', action="store_true", default=False, help='whether do mechanism analysis')
    parser.add_argument('--alpha', type=float, default=0.5, help='input batchsize.')
    parser.add_argument('--show_results', action="store_true", default=False, help='whether show results')
    parser.add_argument('--keywords', type=str, default=None, help='keywords must include in results')
    parser.add_argument('--ban_keywords', type=str, default=None, help='keywords must not include in results')
    parser.add_argument('--results_name', type=str, default=None, help='keywords must include in results')
    parser.add_argument('--model_name_or_path', type=str, default="gpt-3.5-turbo", help='Tasks for instructions generation')
    parser.add_argument('--finetuned_model', type=str, default=None, help='finetuned model path')
    parser.add_argument('--device', type=int, default=9, help='device to use')

    args = parser.parse_args()

    if args.show_results:

        generate_directory_table(args)
        plot_radar(args)

    else:
        
        if args.analysis_behaviour:
            #args.dataset = "analysis_behaviour"
            args.rule_setting = "analysis_behaviour"
            # args.rule_type = "NL"
            args.few_shot = True
            args.cot = True

        elif args.analysis_mechanism:
            args.dataset = "analysis_behaviour"
            args.model_name_or_path = "/netcache/huggingface/llama-2-7b-chat-hf"
            args.rule_setting = "analysis_mechanism"

        finetuned = "finetuned" if args.finetuned_model else "base"
        shot = "few_shot" if args.few_shot else "zero_shot"
        cot = "cot" if args.cot else "no_cot"
        rule_setting = args.rule_setting
        model = args.model_name_or_path.split('/')[-1]
        if args.finetuned_model:
            model += f"_{args.finetuned_model.split('/')[-1]}"
        if args.rule_setting != "no_rule":
            rule_setting += f"_{args.rule_type}"
        args.exp_dir = f"./exp/{args.dataset}/{model}/{rule_setting}/{shot}/{cot}"
        mkdir(args.exp_dir)

        processor = TaskDataProcessor(args)
        processor.load_data()

        if args.use_rcd:
            rcd = RCD(model_name_or_path=args.model_name_or_path, device=args.device)
        else:
            llm = LLM(args)

        args_dict = vars(args)
        output_file = f"{args.exp_dir}/args.json"
        f = open(output_file, "w")
        json.dump(args_dict, f, indent=4)

        do_exp = False
        if args.ignore_exist:
            try:
                f = open(f"{args.exp_dir}/result.json", "r")
                result_data = json.load(f)
                assert result_data
            except:
                do_exp = True
        else:
            do_exp = True            
        
        if do_exp:
        
            if args.analysis_behaviour:
            
                analysis_behaviour(args, processor, llm)
            
            elif args.analysis_mechanism:
            
                analysis_mechanism(args, processor, llm)
            
            elif args.training:
                
                train(args, processor, llm)

            elif args.use_rcd:

                test_rcd(args, processor, rcd)

            else:
                test(args, processor, llm)
                
        else:
            print(f"skip experiment in {args.exp_dir}")