
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
from models import *
# from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataset import *
from tqdm import tqdm
from math import sqrt, log
from executors import HumanevalExecutor, AppsExecutor
from ChatModels import GPT35Chat
import jsonlines
import random
import os
import numpy as np
import re
import astroid
from astroid import nodes
from astroid.builder import AstroidBuilder
import dataclasses
from typing import List, Union, Optional, Literal
from .staticfg import CFGBuilder
import jsonlines
import faiss
import json
from sentence_transformers import SentenceTransformer
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"


MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def trim_header(func_impl):
    if IMPORT_HEADER in func_impl:
        func_impl = func_impl.replace(IMPORT_HEADER, "")
    return func_impl


class KeypointsHoner:
    def __init__(self, args):
        self.args = args
        self.sample_times = 0
        self.gamma = 0.9
        self.save_mid_json = []
        # if 'gpt3.5' in args.arch or 'gpt4' in args.arch:
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.generator = GPT35Chat(args.arch, self.tokenizer, args)
        # else:
            # raise ValueError("Unsupported architecture")

        if args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)
        elif args.dataset == 'apps' or args.dataset == 'codeforces':
            self.executor = AppsExecutor(args)
        else:
            raise ValueError("Unsupported dataset")

        self.term_cond = lambda: self.sample_times > args.max_sample_times
        self.cached_reward = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.cur_buggy_cot = None
        self.cur_explanation = None
        self.cur_corrected_cot = None
        self.cur_vb = None
        self.sample_times = []
        self.st = time.time()

        self.seed_codes = {}

        seed_dir = f"{get_proj_path()}/dataProcess/{self.args.dataset}/seed/{self.args.arch}/"
        difficulty =  self.args.appsdifficulty
        if self.args.dataset == 'apps' or self.args.dataset == 'codeforces':
            seed_path = os.path.join(seed_dir, f"seed_{difficulty}.jsonl")
        else:
            seed_path = os.path.join(seed_dir, f"seed.jsonl")
        with jsonlines.open(seed_path) as reader:
            for item in reader:
                idx = int(item['task_id'].split('/')[1])
                self.seed_codes[idx] = item['solution']
    
        # Prepare sentence transformer model and embeddings
        if args.aug == 'retrieval':
            self.ret = True
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
            explanations = [item["explanation"] for item in self.filtered_debug_pairs]
            self.explanation_embeddings = self.model.encode(explanations, convert_to_tensor=True, normalize_embeddings=True)
        else:
            self.ret = False
            
    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        debug_log = []  
        raw_prompt = problem_instance['prompt'] 

        initial_state = self.tokenizer.encode(raw_prompt)
        if len(initial_state) >= self.args.horizon:
            return None

        cur_pass = 0
        pass_at_k = 1
        is_pass_train = False
        train_reward = 0.0
        test_reward = 0.0
        while cur_pass < pass_at_k and not is_pass_train:
            cur_iter = 0

            if len(self.seed_codes) == 0:
                code_id = self.generator.get_rationale_predicted_sequence(initial_state)
                cur_func_impl = self.tokenizer.decode(code_id)
            else:
                idx = int(problem_instance['task_id'].split('/')[1])
                cur_func_impl = '\n' + self.seed_codes[idx]
                code_id = self.tokenizer.encode(cur_func_impl)

            full_result = self.get_reward(code_id, with_verbal=True)
            complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
            if complete_prog_score >= 1:
                is_pass_train = True
                train_reward = complete_prog_score
                test_reward = self.get_reward(code_id, mode='test')
                break
            
            feedback_str = self.format_feedbacks(self.gather_feedback(verbal_feedbacks))
            current_thoughts = self.think(cur_func_impl)
            messages = []
            max_iters = self.args.max_iters

            while cur_iter < max_iters:
                print('---------------')
                print('cur_iter:', cur_iter)
                print('code: ', cur_func_impl)
                print('---------------')

                # chain_of_thought_with_bugs = self.think(cur_func_impl)
                chain_of_thought_with_bugs = current_thoughts
                print("Generated Chain of Thought (with bugs):", chain_of_thought_with_bugs)

                rethink_result = self.rethink(cur_func_impl, feedback_str, chain_of_thought_with_bugs)
                bug_explanation = rethink_result['bug_explanation']
                correct_chain_of_thought = rethink_result['correct_chain_of_thought']

                print("Bug Explanation:", bug_explanation)
                print("Correct Chain of Thought:", correct_chain_of_thought)
                current_thoughts = correct_chain_of_thought

                self.cur_buggy_cot = chain_of_thought_with_bugs
                self.cur_vb = feedback_str
                self.cur_explanation = bug_explanation
                self.cur_corrected_cot = correct_chain_of_thought

                cur_dict = {
                    'iteration': cur_iter,
                    'initial_program': '\n' + self.seed_codes[int(problem_instance['task_id'].split('/')[1])],
                    'raw_prompt': raw_prompt,
                    'last program': cur_func_impl,
                    'last feedback': self.cur_vb,
                    'detailed_explanation': self.cur_explanation,
                    'buggy_code_cot': self.cur_buggy_cot,
                    'corrected_code_cot': self.cur_corrected_cot,
                }

                generate_code_prompt = (
                    f"You are an expert Python programmer. Below is an algorithmic problem description and a list of thoughts for a correct solution.\n\n"
                    f"Problem Description: {raw_prompt}\n\n"
                    f"Corrected Thoughts: {correct_chain_of_thought}\n\n"
                    f"Your task is to write Python code that implements the logic expressed in these thoughts and solves the problem. "
                    f"Do not include unnecessary comments or explanations, only the code itself."
                )

                new_code_impl = self.generator.generate_response_api(
                    prompt=generate_code_prompt,
                    top_k=1,
                    temperature=0.0,
                    max_length=self.args.horizon,
                )[0]

 

                print("New Generated Code:", new_code_impl)
                cur_dict['corrected_program'] = new_code_impl
                cur_func_impl = new_code_impl

                full_result = self.get_reward(self.tokenizer.encode(cur_func_impl), with_verbal=True)
                complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]
                feedback_str = self.format_feedbacks(self.gather_feedback(verbal_feedbacks))
                cur_dict['train_reward'] = complete_prog_score
                cur_dict['test_reward'] = None
                debug_log.append(cur_dict)

                if complete_prog_score >= 1 or cur_iter == max_iters - 1:
                    is_pass_train = True
                    train_reward = complete_prog_score
                    test_reward = self.get_reward(self.tokenizer.encode(cur_func_impl), mode='test')
                    cur_dict['test_reward'] = test_reward
                    debug_log.append(cur_dict)
                    break

                cur_iter += 1

        complete_programs_ids = list(map(lambda x: list(x), self.cached_reward.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_reward[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]
        best_idx = np.argmax(train_rewards)

        output_dict = {
            'final_program': complete_programs[best_idx],
            'train_reward': train_rewards[best_idx],
            'test_reward': test_rewards[best_idx],
            'all_programs': complete_programs,
            'all_train_rewards': train_rewards,
            'all_test_rewards': test_rewards,
            'avg_sample_time': np.mean(np.array(self.sample_times)),
            'debug_log': debug_log,
        }

        self.cached_reward = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0
        self.save_mid_json = []
        self.generator.save_mid_json = self.save_mid_json
        self.args.rollout_count = -1

        return output_dict

            

    def think(self, cur_func_impl):
        problem_description = self.cur_prob_instance['prompt']
        cur_code_implementation = cur_func_impl

        think_prompt = (
            f"You are an expert Python programmer. Below is an algorithmic question (problem specification) along with the current implementation for solving the problem.\n"
            f"Problem Description: {problem_description}\n"
            f"Current Code Implementation: {cur_code_implementation}\n"
            f"I need you to extract 3 key points (thoughts) that summarize the core algorithm or logic used in this code. "
            f"Please list each thought in a separate entry in the format:\n"
            f"[Thought-i: <summary of key point>]\n"
            f"The output should be a **list of dicts** with each key as `Thought-i`. Do not include explanations or justifications, just focus on capturing the key algorithmic points.\n"
        )
        think_prompt += """
        [
            {"Thought-1": "We could use the print function to finish the task in one line: print(2 + 3)."},
            {"Thought-2": "We should calculate the problem by setting a=2+3, and then print(a)."},
            {"Thought-3": "The problem can't be solved by Python."}
        ]
        """
        insights_analysis = self.generator.generate_response_api(
            prompt=think_prompt, 
            top_k=1, 
            temperature=0.0, 
            max_length=self.args.horizon,
        )[0]
        
        return insights_analysis
    
    def rethink(self, cur_func_impl, feedback_str, chain_of_thought_with_bugs):

        problem_description = self.cur_prob_instance['prompt']

        cur_code_implementation = cur_func_impl

        failed_test = feedback_str
        


        bug_explanation = "No Explanation."
            

        sketch_prompt = (
            f"You are an expert Python programmer. Below is an algorithmic problem description, the current list of thoughts (which contains bugs), "
            f"the current code implementation, and the feedback from running the code.\n\n"
            f"Problem Description: {problem_description}\n\n"
            f"Current Thoughts (with Bugs):\n{chain_of_thought_with_bugs}\n\n"
            f"Current Code Implementation: {cur_code_implementation}\n\n"
            f"Feedback from the code execution: {failed_test}\n\n"
            f"Please analyze and generate 1-2 new thoughts to correct the approach.\n"
            f"**Format Requirements:**\n"
            f"1. Output ONLY a JSON-parsable list of dictionaries\n"
            f"2. Each dictionary must use 'Thought-i' as key (i continues from previous sequence)\n"
            f"3. Each value should concisely state one algorithmic insight\n"
            f"Example:\n"
            f'[\n  {{"Thought-4": "Using depth-first search for tree traversal"}},\n  {{"Thought-5": "Handling leaf nodes with null checks"}}\n]'
        )

        # Generate and parse response
        generated_response = self.generator.generate_response_api(
            prompt=sketch_prompt,
            temperature=0.0,
            top_k=1,
            max_length=self.args.horizon,
        )[0]

        # Parse both existing and new thoughts
        try:
            existing_thoughts = json.loads(chain_of_thought_with_bugs)
            new_thoughts = json.loads(generated_response)
            merged_thoughts = existing_thoughts + new_thoughts
            correct_chain_of_thought = json.dumps(merged_thoughts)
        except json.JSONDecodeError:
            print("ERROR!!!!!")
            correct_chain_of_thought = chain_of_thought_with_bugs  # Fallback

        return {
            "bug_explanation": bug_explanation,
            "correct_chain_of_thought": correct_chain_of_thought
        }

            
    def get_reward(self, s, mode='train', with_verbal=False):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            if with_verbal:
                return [self.cached_reward[tuple(s)], self.cached_verbal_feedback[tuple(s)]]
            else:
                return self.cached_reward[tuple(s)]


        output_str = self.convert_state_to_program(s)


        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, output_str, mode, with_verbal=with_verbal)  # with_verbal: curr_res=[[True/False, feedback_dict]]
            fixed = []
            verbal_feedbacks = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                if with_verbal:
                    verbal_feedbacks.append(e[1])
                    e = e[0]
                fixed.append(e)

            curr_res = fixed
            # if not np.all(curr_res):
            #     print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            curr_res = []

        # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
        assert isinstance(curr_res, list)
        pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
        reward = pass_rate


        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            if with_verbal:
                self.cached_verbal_feedback[tuple(s)] = verbal_feedbacks

        if with_verbal:
            return [reward, verbal_feedbacks]
        else:
            return reward
        
    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s
    
    def format_feedbacks(self, verbal_feedbacks):

        feedback_str = ""
        tmp_count = 0
        

        feedbacks_to_process = verbal_feedbacks
        
        for k, feedback in enumerate(feedbacks_to_process):
            if not isinstance(feedback, str):
                if len(self.tokenizer.encode(feedback['output'])) > 2048:
                    tmp_shorter = self.tokenizer.encode(feedback['output'])[:2048]
                    feedback['output'] = self.tokenizer.decode(tmp_shorter)
                feedback_str += f"\n\n# Failed test {tmp_count + 1}: {feedback['output']}"
                tmp_count += 1
        return feedback_str
    
    
    def gather_feedback(self, verbal_feedbacks):



        false_feedback = [item for item in verbal_feedbacks if isinstance(item, dict)]
        ori_length = len(false_feedback)
        tmp_feedback = [item for item in false_feedback if len(self.tokenizer.encode(item['output'])) < 512]
        tmp_feedback = tmp_feedback[:5]

        if len(tmp_feedback) == 0 and ori_length > 0:

            if self.args.dataset == 'apps':
                with_ldb = False
            false_feedback = false_feedback[:5]
            tmp_feedback = [
                {
                    'error': item['error'],
                    'output': shorten_text_block(item['output'], self.tokenizer, total_length=512,
                                                 section_length=512)
                }
                for item in false_feedback
            ]
        false_feedback = tmp_feedback
        return false_feedback
        
        
def shorten_text_block(full_text, tokenizer, total_length=512, section_length=512, front_tokens=100, end_tokens=100):

    # 分割文本为三部分
    try:
        sections = {
            "Input": full_text.split("# Input:")[1].split("# Ground Truth Output:")[0],
            "Ground Truth": full_text.split("# Ground Truth Output:")[1].split("# Current Execution Output:")[0],
            "Execution Output": full_text.split("# Current Execution Output:")[1],
        }
    except Exception as e:
        return tokenizer.decode(tokenizer.encode(full_text)[:total_length])

    for key in sections:
        if len(tokenizer.encode(sections[key])) > section_length:
            sections[key] = (tokenizer.decode(tokenizer.encode(sections[key])[:front_tokens]) +
                             " ... " + tokenizer.decode(tokenizer.encode(sections[key])[-end_tokens:]))


    formatted_text = (
        f"# Input:{sections['Input']}"
        f"# Ground Truth Output:{sections['Ground Truth']}"
        f"# Current Execution Output:{sections['Execution Output']}"
    )
    return formatted_text