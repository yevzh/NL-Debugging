

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


class CoTHoner:
    def __init__(self, args):
        self.args = args
        self.sample_times = 0
        self.gamma = 0.9
        self.save_mid_json = []
        # if 'gpt3.5' in args.arch or 'gpt4' in args.arch:
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.generator = GPT35Chat(args.arch, self.tokenizer, args)
        self.gptgenerator = GPT35Chat(args.arch, self.tokenizer, args)
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

        # Load seed programs if applicable
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
                # print(idx)
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
                print(idx)
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
            messages = []
            max_iters = self.args.max_iters
            current_cot = self.think(cur_func_impl)
            while cur_iter < max_iters:
                print('---------------')
                print('cur_iter:', cur_iter)
                print('code: ', cur_func_impl)
                print('---------------')
                if self.args.continuous:
                    chain_of_thought_with_bugs = current_cot
                else:
                    chain_of_thought_with_bugs = self.think(cur_func_impl)
                print("Generated Chain of Thought (with bugs):", chain_of_thought_with_bugs)

                rethink_result = self.rethink(cur_func_impl, feedback_str, chain_of_thought_with_bugs)
                bug_explanation = rethink_result['bug_explanation']
                correct_chain_of_thought = rethink_result['correct_chain_of_thought']
                current_cot = correct_chain_of_thought
                print("Bug Explanation:", bug_explanation)
                print("Correct Chain of Thought:", correct_chain_of_thought)

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
                    f"You are an expert Python programmer. Below is a high-level natural language sketch for a correct solution to an algorithmic problem, "
                    f"along with the problem description.\n\n"
                    f"Problem Description: {raw_prompt}\n\n"
                    f"Correct sketch: {correct_chain_of_thought}\n\n"
                    f"Your task is to write Python code that implements this sketch and solves the problem. "
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
        print(problem_description)
        print(cur_code_implementation)

        think_prompt = (
            f"You are an expert Python programmer. Below is an algorithmic question (problem specification) along with the current implementation for solving the problem.\n"
            f"Problem Description: {problem_description}\n"
            f"Current Code Implementation: {cur_code_implementation}\n"
            f"Your task is to generate a **Natural Language Sketch** for this code. "
            f"This sketch should describe the logical reasoning or steps that the code is trying to follow in order to solve the problem. "
            f"Do not focus on syntax or specific code lines, but explain the thought process or approach the code takes to solve the problem at a high level."
        )
        
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


        

        rethink_prompt = (
            f"You are an expert Python programmer. You will be provided with an algorithmic problem description, "
            f"the current Python code implementation, and the execution feedback that indicates where the code went wrong.\n\n"
            f"Problem Description: {problem_description}\n\n"
            f"Current Code Implementation: {cur_code_implementation}\n\n"
            f"The code has some bugs. Below is the feedback from executing the code:\n"
            f"{failed_test}\n\n"
            f"Here is the buggy natural language sketch for the code implementation: \n{chain_of_thought_with_bugs}\n"
            f"Please analyze the feedback and provide an explanation of what went wrong in the code and why it failed in this sketch. "
            f"Do not provide specific steps to fix the sketch. Focus solely on explaining the root cause of the issue in two or three sentences."
        )

        bug_explanation = self.generator.generate_response_api(
            prompt=rethink_prompt,
            temperature=0.0,
            top_k=1,
            max_length=self.args.horizon,
        )[0]
        

 
        sketch_prompt = (
            f"You are an expert Python programmer. Below is an algorithmic problem description, the current natural language sketch of the solution (which contains bugs), "
            f"the current code implementation, and the feedback from running the code, as well as a detailed expert analysis of the bug.\n\n\n\n"
            f"Problem Description: {problem_description}\n\n"
            f"Current Natural Language Sketch (with Bugs): \n{chain_of_thought_with_bugs}\n\n"
            f"Current Code Implementation: \n{cur_code_implementation}\n\n"
            f"Feedback from the code execution: \n{failed_test}\n\n"
            f"Bug Analysis: \n{bug_explanation}\n\n"
            f"Please rethink the approach and generate a new, corrected sketch for solving the problem referring to the bug explanation for this sketch. "
            f"This new sketch should outline the correct reasoning and steps needed to solve the problem, avoiding the issues found in the previous sketch. "
            f"Do not provide specific code, just a high-level step-by-step natural language explanation of the corrected approach."
        )


        correct_chain_of_thought = self.generator.generate_response_api(
            prompt=sketch_prompt,
            temperature=0.0,
            top_k=1,
            max_length=self.args.horizon,
        )[0]    
        
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