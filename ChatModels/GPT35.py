# -*- coding:utf-8 _*-
import torch
from typing import List, Union, Optional, Literal
import dataclasses
import transformers
from utils import get_raw_data_path
from openai import OpenAI
import openai
import tiktoken
import time
from .cache import GPTTopKCache, GPTSeqCache
import math
import re
import json
import os
from time import sleep
from vllm import LLM, SamplingParams


def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.encode(l, allowed_special={'<|endoftext|>'}))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.encode(messages[0].content, allowed_special={'<|endoftext|>'}))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages


class GPT35Chat:
    def __init__(self, model_name, tokenizer, args, save_mid_json=[]):
        self.name = model_name
        if model_name in ['gpt3.5', 'kimi', 'gpt4', 'gpt4o-mini', 'gpt4o', 'o1-preview', 'o1-mini', 'claude']:
            self.call_mode = 'api'
        else:
            if args.aliyun:
                self.call_mode = 'api'
            else:
                self.call_mode = 'local'
            
        # if model_name == 'localapi':
        #     self.call_mode = 'localapi'
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI()
        if self.call_mode == 'api':
            if os.getenv("OPENAI_API_KEY") == "sk-xxxx":
                self.client.base_url = "http://open.xiaoai.one/v1"

        elif self.call_mode == 'local':
            self.client = OpenAI(
                base_url=f"http://localhost:{self.args.apiport}/v1",
                api_key="token-abc123"
            )
        if self.call_mode == 'localapi':
            self.client = OpenAI(
                base_url = "http://localhost:8000/v1",
                api_key = "token-abc123",
            )
        # print(self.client.models.list())
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.save_mid_json = save_mid_json
        if self.name == "deepseek":
            self.load_model_dir = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            
        # print(self.load_model_dir)
        if self.call_mode == 'local' and self.args.local_callmode== 'load':
            self.llm = LLM(
                model=self.load_model_dir,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=17888,
            )
            self.llm_tokenizer = self.llm.get_tokenizer()
        self.log_prob_provided = False
        if self.name == 'gpt3.5':
            self.model_name = 'gpt-3.5-turbo-0125'
        elif self.name == 'gpt4':
            self.model_name = 'gpt-4-turbo-2024-04-09'
        elif self.name == 'gpt4o-mini':
            self.model_name = 'gpt-4o-mini'
        elif self.name == 'gpt4o':
            self.model_name = 'gpt-4o'
        elif self.name == 'o1-preview':
            self.model_name = 'o1-preview'
        elif self.name == 'o1-mini':
            self.model_name = 'o1-mini'
        elif self.name == 'claude':
            self.model_name = 'claude-3-5-sonnet-20241022'
        elif self.name == 'deepseek':
            self.model_name = 'DeepSeek-Coder-V2-Lite-Instruct'
        else:
            print(f'Model {self.name} not implemented error!')
            assert 0
        if self.args.aliyun:
            self.model_name = 'deepseek-v3'
        
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.save_mid_json = save_mid_json

        
    def generate_chat(self, messages, stop, max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1):
        for ti in range(20):  # try multiple times
            sleep_interval = 7
            if self.call_mode == 'api':
                try:
                    new_messages = change_messages(self.tokenizer, messages, 8000)
                    messages = new_messages
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[dataclasses.asdict(message) for message in messages],
                        temperature=temperature,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        n=num_comps,
                        stop=stop
                    )
                except Exception as e:
                    print("GPT Error:", str(e))
                    if "context_length_exceeded" in str(e):
                        messages = change_messages(self.tokenizer, messages, 8000)
                        continue
                    else:
                        sleep_t = sleep_interval * (ti + 1)
                        print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                        with open("error.log", "a") as f:
                            f.write(f"gpt failed multiple times with: {str(e)}\n")
                        sleep(sleep_t)
                        continue

                input_token_num = 0
                for msg in messages:
                    input_token_num += len(self.tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(response.choices[0].message.content, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
                return response.choices[0].message.content  # type: ignore
            else:
                if self.args.local_callmode == 'load':
                    new_messages = change_messages(self.tokenizer, messages, 8000)
                    messages = new_messages
                    prompt = self.llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True,  tokenize=False)
                    if self.name == 'deepseek-coder-33b-instruct':
                        stop_token_ids = [self.llm_tokenizer.eos_token_id, 32021]  # 32021 is <EOT>
                    else:
                        stop_token_ids = [self.llm_tokenizer.eos_token_id]
                    outputs = self.llm.generate(
                        prompts=prompt,
                        sampling_params=SamplingParams(
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_k=1,
                            stop_token_ids=stop_token_ids,
                        )
                    )
                    response_text = outputs[0].outputs[0].text

                    input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                    output_token_num = len(self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'}))
                    self.args.total_input_token_num += input_token_num
                    self.args.total_output_token_num += output_token_num
                    return response_text
                else:
                    try:
                        if self.args.apiport == 12343:
                            self.model_name = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
                        new_messages = change_messages(self.tokenizer, messages, 8000)
                        messages = new_messages
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[dataclasses.asdict(message) for message in messages],
                            temperature=temperature,
                            top_p=1,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            n=num_comps,
                            stop=stop
                        )
                    except Exception as e:
                        print("GPT Error:", str(e))
                        if "context_length_exceeded" in str(e):
                            messages = change_messages(self.tokenizer, messages, 8000)
                            continue
                        else:
                            sleep_t = sleep_interval * (ti + 1)
                            print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                            with open("error.log", "a") as f:
                                f.write(f"gpt failed multiple times with: {str(e)}\n")
                            sleep(sleep_t)
                            continue

                    input_token_num = 0
                    for msg in messages:
                        input_token_num += len(self.tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
                    output_token_num = len(self.tokenizer.encode(response.choices[0].message.content, allowed_special={'<|endoftext|>'}))
                    self.args.total_input_token_num += input_token_num
                    self.args.total_output_token_num += output_token_num
                    return response.choices[0].message.content  # type: ignore
        else:
            print(f'try failure with multiple times')
            assert False

    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None, temperature=0.0):
        sys_msg = "You are a helpful code generator that generate code to complete the given problem."
        if system_message:
            sys_msg = system_message
        for ti in range(20):
            sleep_interval = 7
            try:
                if self.call_mode == 'api':
                    if not self.log_prob_provided:
                        top_k = None
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length, 
                        temperature=temperature,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=self.log_prob_provided,
                        top_logprobs=top_k
                    )
                    message = response.choices[0].message.content
                    if self.log_prob_provided:
                        log_prob = response.choices[0].logprobs.content 
                    else:
                        log_prob = []
                elif self.call_mode == 'local':
                    if self.args.local_callmode == 'load':
                        messages = [{"role": "system", "content": f'{sys_msg}\n'},
                                    {"role": "user", "content": prompt}]
                        prompt = self.llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                        if self.name == 'deepseek-coder-33b-instruct':
                            stop_token_ids = [self.llm_tokenizer.eos_token_id, 32021] # 32021 is <EOT>
                        else:
                            stop_token_ids = [self.llm_tokenizer.eos_token_id]
                        outputs = self.llm.generate(
                            prompts=prompt,
                            sampling_params=SamplingParams(
                                temperature=temperature,
                                max_tokens=max_length,
                                top_k=1,
                                stop_token_ids=stop_token_ids,
                            )
                        )
                        message = outputs[0].outputs[0].text
                        log_prob = []
                    elif self.args.local_callmode == 'api':
                        if not self.log_prob_provided:
                            top_k = None
                        models = self.client.models.list()
                        # print(models)
                        if self.args.apiport == 12343:
                            self.model_name = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                            max_tokens=max_length,  
                            temperature=temperature,
                            frequency_penalty=0,
                            presence_penalty=0,
                            logprobs=self.log_prob_provided,
                            top_logprobs=top_k
                        )
                        message = response.choices[0].message.content
                        if self.log_prob_provided:
                            log_prob = response.choices[0].logprobs.content  
                            log_prob = []

                input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(message, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:8000])
                    continue
                else:
                    print("GPT Error:", str(e))
                    sleep_t = sleep_interval * (ti + 1)
                    print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                    with open("error.log", "a") as f:
                        f.write(f"gpt failed multiple times with: {str(e)}\n")
                    sleep(sleep_t)
                    continue
            return message, log_prob
        else:
            print(f'try failure with multiple times')
            assert False

