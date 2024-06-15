datasets=("clutrr" "salad" "law" "deer" "theoremQA" "ulogic")
models=("vllm_Llama-2-7b-chat-hf" "vllm_Meta-Llama-3-8B" "vllm_Mistral-7B-Instruct-v0.2" "gpt-3.5-turbo" "gpt-4-turbo" "gpt-4o" "vllm_Phi-3-small-8k-instruct" "vllm_Yi-1.5-6B-Chat")

pids=()

function kill_processes {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    exit
}

trap kill_processes INT

# exp 1&2
rule_settings=("no_rule" "golden_rule" "few_rule" "all_rule")
rule_types=("NL" "FOL")

for dataset in "${datasets[@]}"; do
    for rule_setting in "${rule_settings[@]}"; do
        for rule_type in "${rule_types[@]}"; do
            for model in "${models[@]}"; do
                python main.py --dataset "$dataset" --rule_setting "$rule_setting" --rule_type "$rule_type" --model_name_or_path "$model" &
                pids+=($!)
            done
        done
    done
done

# exp 3
cots=("cot" "")
rule_settings=("golden_rule" "few_rule")
for dataset in "${datasets[@]}"; do
    for rule_setting in "${rule_settings[@]}"; do
        for cot in "${cots[@]}"; do
            for model in "${models[@]}"; do
                if [[ "$cot" ]]; then
                    python main.py --dataset "$dataset" --rule_setting "$rule_setting" --rule_type "NL" --model_name_or_path "$model" --few_shot --cot &
                else
                    python main.py --dataset "$dataset" --rule_setting "$rule_setting" --rule_type "NL" --model_name_or_path "$model" --few_shot &
                fi
                pids+=($!)
            done
        done
    done
done

# exp 4 注意是否存在对应的反事实数据集
rule_settings=("golden_rule" "few_rule")
for dataset in "${datasets[@]}"; do
    dataset_cf="${dataset}-cf"
    for rule_setting in "${rule_settings[@]}"; do
        for model in "${models[@]}"; do
            python main.py --dataset "$dataset_cf" --few_shot --cot --rule_setting "$rule_setting" --rule_type "NL" --model_name_or_path "$model" &
            pids+=($!)
        done
    done
done

# exp 5 注意是否存在对应的反事实数据集
for dataset in "${datasets[@]}"; do
    dataset_cf="${dataset}-cf"
    for model in "${models[@]}"; do
        python main.py --dataset "$dataset_cf" --model_name_or_path "$model" --analysis_behaviour &
        pids+=($!)
    done
done

wait

python main.py --show_results
python main.py --show_results --keywords exp1 --results_name exp1
python main.py --show_results --keywords exp2 --results_name exp2
python main.py --show_results --keywords exp3 --results_name exp3
python main.py --show_results --keywords exp4 --results_name exp4
python main.py --show_results --keywords exp5 --results_name exp5