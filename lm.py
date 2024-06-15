import os

import httpx
import openai
import numpy as np
import torch
import math
import copy
from typing import Union, List, Optional
from collections import Counter
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch.nn.functional as F
from model import load_base_model_and_tokenizer, load_finetuned_model_and_tokenizer
from utils import StoppingCriteriaList, StopAtSpecificTokenCriteria
from openai import OpenAI

class LLM():
    
    def __init__(self, args):
        
        self.device = args.device
        self.max_token = args.max_token
        self.finetuned_model = args.finetuned_model
        self.model_name_or_path = args.model_name_or_path
        if type(args.model_name_or_path) == str:
            if args.model_name_or_path in ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
                self.model = args.model_name_or_path
                if os.getenv("proxy", None):
                    self.openai = OpenAI(
                        api_key= os.getenv("OPENAI_API_KEY"),
                        http_client=httpx.Client(
                            proxies=os.getenv("proxy", None),
                            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                        ),
                    )
                else:
                    self.openai = OpenAI(
                        api_key= os.getenv("OPENAI_API_KEY")
                    )

            elif "vllm" in args.model_name_or_path:
                openai_api_key = "hhhhhhhhhh"
                openai.proxy = os.getenv("proxy", None)
                # deploy your vllm server
                if "Llama-2-7b-chat-hf" in args.model_name_or_path:
                    openai_api_base = None
                elif "Meta-Llama-3-8B" in args.model_name_or_path:
                    openai_api_base = None
                elif "Mistral-7B-Instruct-v0.2" in args.model_name_or_path:
                    openai_api_base = None
                elif "Yi-1.5-6B-Chat" in args.model_name_or_path:
                    openai_api_base = None
                elif "Phi-3-small-8k-instruct" in args.model_name_or_path:
                    openai_api_base = None
                else:
                    raise ValueError("model not deployed yet")
                self.client = OpenAI(
                            api_key=openai_api_key,
                            base_url=openai_api_base,
                        )
                models = self.client.models.list()
                self.model = models.data[0].id
                print(f"using vllm server, the model name used now is {self.model}!!!!")
            else:
                if self.finetuned_model:
                    self.model, self.tokenizer = load_finetuned_model_and_tokenizer(args.model_name_or_path, args.finetuned_model)
                    self.ending_idx = self.tokenizer.eos_token_id
                    self.padding_idx = self.tokenizer.pad_token_id
                else:
                    self.model, self.tokenizer = load_base_model_and_tokenizer(args.model_name_or_path)
                    self.ending_idx = self.tokenizer.eos_token_id
                    self.padding_idx = self.tokenizer.pad_token_id
        else:
            self.model, self.tokenizer = args.model_name_or_path
            self.ending_idx = self.tokenizer.eos_token_id
            self.padding_idx = self.tokenizer.pad_token_id
        self.args = args

    def __call__(self, prompt, stop=[], temperature=0, do_sample=False, top_p=1):
    
        response = None
        if self.model in ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
            response = self.openai_api(prompt, stop, temperature, top_p)
        elif "vllm" in self.model_name_or_path:
            response = self.vllm(prompt, stop, temperature, top_p)
        else:
            response = self.local(prompt, stop, do_sample=do_sample)

        return response
    
    def local(self, prompts, stop, do_sample):
        
        with torch.no_grad():
            token_id_list = [self.tokenizer.encode(s)[-1] for s in stop]
            #token_id_list.append(self.ending_idx)
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list))
        
            if type(prompts) == str:

                messages = [{"role": "user", "content": prompts}]  
                input_prompts = self.tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = self.tokenizer.encode(input_prompts, return_tensors='pt').to(self.device)
                response = self.model.generate(input_ids=inputs, 
                                                max_length=inputs.shape[1]+self.max_token, 
                                                early_stopping=True,
                                                eos_token_id=self.ending_idx,
                                                pad_token_id=self.padding_idx,
                                                do_sample=do_sample,
                                                stopping_criteria=stopping_criteria
                                                )
                
                response = response[0][inputs.shape[1]:]
                response_text = self.tokenizer.decode(response).strip('\n')
                return response_text
        
            else:
                messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
                input_prompts = [self.tokenizer.apply_chat_template(message, tokenize=False) for message in messages]
                inputs =  self.tokenizer(input_prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device) 
                response = self.model.generate(input_ids=inputs, 
                                                max_length=inputs.shape[1]+self.max_token, 
                                                early_stopping=True,
                                                eos_token_id=self.ending_idx,
                                                pad_token_id=self.padding_idx,
                                                do_sample=do_sample,
                                                stopping_criteria=stopping_criteria
                                                )
                response_text = []        
                for i in range(len(prompts)):
                    response_i = response[i][inputs.shape[1]:]
                    response_text.append(self.tokenizer.decode(response_i).strip("\n").replace("<s>", "").replace("</s>", ""))
                return response_text
    
    def openai_api(self, prompts, stop=["\n"], temperature=0, top_p=0):
        
        if type(prompts) == str:
        
            messages = [{"role": "user", "content": prompts}]
            model_id = self.finetuned_model if self.finetuned_model else self.model
            response = self.openai.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_token,
                    top_p=top_p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
            response = response.dict()
            return response["choices"][0]["message"]["content"]
        
        else:
            
            responses = []

            for prompt in prompts:

                messages = [{"role": "user", "content": prompt}]
                model_id = self.finetuned_model if self.finetuned_model else self.model
                response = self.openai.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_token,
                        top_p=top_p,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=stop
                    )
                response=response.dict()
                responses.append(response["choices"][0]["message"]["content"])

            return responses

    def vllm(self, prompts, stop, temperature=0, top_p=0):
        if type(prompts) == str:

            messages = [{"role": "user", "content": prompts}]
            response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_token,
                    top_p=top_p,
                    stop=["<|eot_id|>"]
                )
            return  response.choices[0].message.content

        else:

            responses = []

            for prompt in prompts:
                import re
                success = False
                try_ix = 0
                while not success:
                    try_ix += 1
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=self.max_token,
                            top_p=top_p,
                            stop=["<|eot_id|>"]
                        )
                    pattern = {"theoremqa"[:6]: r"Answer: ([abcd])", "ulogic": r"Answer: (.*?)\."}.get(self.args.dataset.lower()[:6], "")
                    success = bool(re.search(pattern, response.choices[0].message.content))
                    if not success:
                        temperature = 0.5
                        top_p = 1
                        print(f"{response.choices[0].message.content}\nresponse not success parse with try {try_ix}.")
                        if try_ix >= 5:
                            break
                responses.append(response.choices[0].message.content)
            return responses

    def lm_loss(self, prompt, target):
        prompt_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        total_inputs = self.tokenizer.encode(prompt+target, return_tensors='pt').to(self.device)
        target_inputs = self.tokenizer.encode(target, return_tensors='pt').to(self.device)

        # 直接解码target文本的损失
        with torch.no_grad():
            target_loss = self.model(target_inputs, labels=target_inputs)[0]
            # logits = self.model(total_inputs)[0]
            # loss_fct = torch.CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
            
            prompt_loss = self.model(prompt_inputs, labels=prompt_inputs)[0]
            total_loss = self.model(total_inputs, labels=total_inputs)[0]
            
            target_loss_given_prompt = (total_loss*(total_inputs.shape[1]-1)-prompt_loss*(prompt_inputs.shape[1]-1))/(target_inputs.shape[1]-1)
        
        return target_loss.tolist(), target_loss_given_prompt.tolist()

    def lm_loss_on_targets(self, prompts, targets):
        
        with torch.no_grad():
            losses = []
            prompt_inputs =  self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device) 

            for target in targets:
                totals = [prompt + target for prompt in prompts]
                total_inputs =  self.tokenizer(totals, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device) 
                
                labels = self.create_lm_labels(prompt_inputs, total_inputs)
                
                logits = self.model(total_inputs, labels=labels)[1]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss.view(len(prompts),-1).sum(dim=1)
                
                losses.append(loss)
                
            return losses
        
    def create_lm_labels(self, prompt_inputs, total_inputs):
        
        labels = copy.deepcopy(total_inputs)
        
        # if prompt_inputs.shape[1] == total_inputs.shape[1]:
            
        #     labels[:, :-1] = -100
            
        #     return labels
        
        # else:
        
        #     labels[:, :prompt_inputs.shape[1]] = -100
        
        labels[:, :prompt_inputs.shape[1]-1] = -100
            
        return labels
            # for i in range(prompt_inputs.shape[0]):
            
            #     for j in range(total_inputs.shape[1]):
                    
            #         if total_inputs[i][j] != prompt_inputs[i][j]:

def post_process(response):
    processed_response = response.rstrip('\n.')
    return processed_response

class RCD:
    
    def __init__(self, model_name_or_path: str, device: Union[int,str] = 0):
        self.model, self.tokenizer = load_base_model_and_tokenizer(model_name_or_path)


    def _top_p_sampling(self,
                        logits: torch.Tensor,
                        top_p: float = 0.9,
                        filter_value: float = -float("Inf"),
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p

        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

        return logits


    def _top_k_sampling(self,
                        logits: torch.Tensor,
                        top_k: int = 20,
                        filter_value: float = -float("Inf"),
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # * logit 값이 Top-k의 토큰 중 가장 작은 값보다 작은 토큰의 인덱스 반환
        logits[indices_to_remove] = filter_value

        return logits


    def predict_next_token(self,
                           logits: torch.Tensor,
                           decoding_strategy: str,
                           top_p: float,
                           top_k: int,
                           use_repetition_penalty: bool,
                           repetition_penalty_value: float,
                           generated_tokens: List[set] = None
                           ) -> torch.Tensor :

        # * Repetitin Penalty 참고 코드 : https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.enforce_repetition_penalty_
        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0) # generated_tokens에 있는 토큰들은 penalty를 repetition_penalty_value로, 없는 토큰들은 1.0(현상 유지)으로 설정
            logits *= torch.where(logits < 0, penalty, 1.0/penalty) # if logit is smaller than 0, multiply with penalty, else divide by penalty

        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            logits = self._top_p_sampling(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            logits = self._top_k_sampling(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1)

        return next_token


    def generate(self,
                input_texts: List[tuple],
                alpha: float = 0.5,
                max_length: int = 100,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False,
                repetition_penalty_value: float = 1.0,
                ) -> List[List[int]]:

        inputs_positive = [self.tokenizer.apply_chat_template([{"role": "user", "content": i[0]}], tokenize=False) for i in input_texts]
        inputs_negative = [self.tokenizer.apply_chat_template([{"role": "user", "content": i[1]}], tokenize=False) for i in input_texts]
        inputs_positive = [i[0] for i in input_texts]
        inputs_negative = [i[1] for i in input_texts]
        tokens_positive = self.tokenizer(inputs_positive, return_tensors="pt", padding=True, truncation=True)
        ids_positive = tokens_positive['input_ids']
        attention_mask_positive = tokens_positive['attention_mask']
        tokens_negative = self.tokenizer(inputs_negative, return_tensors="pt", padding=True, truncation=True)
        ids_negative = tokens_negative['input_ids']
        attention_mask_negative = tokens_negative['attention_mask']
        
        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_texts)
        unfinished_sents = ids_positive.new(batch_size).fill_(1)
        sent_lengths = ids_positive.new(batch_size).fill_(max_length)

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]

        # Generate tokens
        with torch.no_grad():
            while cur_len < max_length:

                outputs_positive = self.model(ids_positive, attention_mask=attention_mask_positive)
                next_token_logits_positive = outputs_positive.logits[:, -1, :]

                outputs_negative = self.model(ids_negative, attention_mask=attention_mask_negative)
                next_token_logits_negative = outputs_negative.logits[:, -1, :]
                
                next_token_logits = (1 + alpha) * next_token_logits_positive - alpha * next_token_logits_negative

                # Predict next token according to decoding strategy
                next_token = self.predict_next_token(logits=next_token_logits,
                                                    decoding_strategy=decoding_strategy,
                                                    top_p=top_p_value,
                                                    top_k=top_k_value,
                                                    use_repetition_penalty=use_repetition_penalty,
                                                    repetition_penalty_value=repetition_penalty_value,
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                # Handle EOS token and padding
                if self.tokenizer.eos_token_id is not None:
                    tokens_to_add = next_token * unfinished_sents + (self.tokenizer.pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # Update input_ids and attention masks for the next forward pass
                ids_positive = torch.cat([ids_positive, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_positive = torch.cat([attention_mask_positive, unfinished_sents.unsqueeze(-1)], dim=-1)
                ids_negative = torch.cat([ids_negative, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_negative = torch.cat([attention_mask_negative, unfinished_sents.unsqueeze(-1)], dim=-1)

                cur_len += 1

                # Update generated tokens and check for completion
                for i, token in enumerate(tokens_to_add.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)

                # Check for sentences that are finished
                if self.tokenizer.eos_token_id is not None:
                    eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        # Return the generated tokens
        
        response_text = []        
        for i in range(batch_size):
            response_i = generated_tokens[i]
            response_text.append(self.tokenizer.decode(response_i).strip("\n").replace("<s>", "").replace("</s>", ""))
        return response_text


    def generate_backup(self,
                input_texts: List[str],
                contexts: Optional[List[str]] = None,
                use_context_aware: bool = True,
                alpha: float = 0.5,
                max_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False,
                repetition_penalty_value: float = 1.0,
                ) -> List[List[int]]:

        # Tokenize 'input_texts' and create attention masks
        tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        # Tokenize 'contexts' after concatenating with 'input_ids' if 'contexts' is not None
        if contexts and use_context_aware:
            inputs_with_contexts = [context + self.tokenizer.eos_token + input_text for context, input_text in zip(contexts, input_texts)]
            tokenized_inputs_with_contexts = self.tokenizer(inputs_with_contexts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids_with_contexts = tokenized_inputs_with_contexts['input_ids']
            attention_mask_with_contexts = tokenized_inputs_with_contexts['attention_mask']
        else:
            input_ids_with_contexts = input_ids
            attention_mask_with_contexts = attention_mask

        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids_with_contexts.new(batch_size).fill_(1)
        sent_lengths = input_ids_with_contexts.new(batch_size).fill_(max_length)

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]

        # Generate tokens
        with torch.no_grad():
            while cur_len < max_length:

                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)

                # * Context-aware Decoding
                if contexts and use_context_aware:
                    outputs_with_contexts = self.model(input_ids_with_contexts, attention_mask=attention_mask_with_contexts)
                    next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                    next_token_logits = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits

                # Predict next token according to decoding strategy
                next_token = self.predict_next_token(logits=next_token_logits,
                                                    decoding_strategy=decoding_strategy,
                                                    top_p=top_p_value,
                                                    top_k=top_k_value,
                                                    use_repetition_penalty=use_repetition_penalty,
                                                    repetition_penalty_value=repetition_penalty_value,
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                # Handle EOS token and padding
                if self.tokenizer.eos_token_id is not None:
                    tokens_to_add = next_token * unfinished_sents + (self.tokenizer.pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # Update input_ids and attention masks for the next forward pass
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, unfinished_sents.unsqueeze(-1)], dim=-1)
                input_ids_with_contexts = torch.cat([input_ids_with_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_with_contexts = torch.cat([attention_mask_with_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)

                cur_len += 1

                # Update generated tokens and check for completion
                for i, token in enumerate(tokens_to_add.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)

                # Check for sentences that are finished
                if self.tokenizer.eos_token_id is not None:
                    eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        # Return the generated tokens
        return generated_tokens
